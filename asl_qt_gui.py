"""
ASL Sign Language Recognition - Raspberry Pi PyQt5 (simplified)

Layout (designed for ~800x480 small screens, fullscreen):
  +-------------------------------------------------+
  |                                                 |
  |         [ BIG CAMERA PREVIEW ]                  |
  |                                                 |
  |   ┌──────────── overlays ───────────┐           |
  |   │ WAITING                FPS 12.3 │           |
  |   └─────────────────────────────────┘           |
  |                                                 |
  |   ┌──── HELLO  87% ────┐                        |
  |   └────────────────────┘                        |
  |                                                 |
  |   sentence: hello mom drink                     |
  |   [Backspace] [Clear] [Quit]                    |
  +-------------------------------------------------+
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout
)

from ai_edge_litert.interpreter import Interpreter


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8
HANDS_LOST_FRAMES = 15
LH_START, LH_END = 468, 489
RH_START, RH_END = 522, 543
CONFIDENCE_THRESHOLD = 0.30


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class SignClassifier:
    def __init__(self, model_path, label_map_path, num_threads=4):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(label_map_path, encoding="utf-8") as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        print(f"[Model] Loaded {len(self.idx_to_sign)} sign classes")

    def predict(self, sequence, top_k=3):
        if sequence.shape[0] > MAX_LEN:
            start = (sequence.shape[0] - MAX_LEN) // 2
            sequence = sequence[start:start + MAX_LEN]

        self.interpreter.resize_tensor_input(
            self.input_details[0]["index"], list(sequence.shape))
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(
            self.input_details[0]["index"], sequence.astype(np.float32))
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(
            self.output_details[0]["index"]).flatten()

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(self.idx_to_sign[int(i)], float(probs[i])) for i in top_idx]


def extract_landmarks(results):
    out = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    if results.face_landmarks:
        out[0:468] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
            dtype=np.float32)
    if results.left_hand_landmarks:
        out[LH_START:LH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32)
    if results.pose_landmarks:
        out[489:522] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32)
    if results.right_hand_landmarks:
        out[RH_START:RH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32)
    return out


def hands_visible(results):
    return (results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None)


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------
class InferenceThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    state_changed = pyqtSignal(str)
    fps_updated   = pyqtSignal(float)
    result_ready  = pyqtSignal(list, int)   # predictions, n_frames

    def __init__(self, classifier, complexity=1, width=640, height=480, rotate=0):
        super().__init__()
        self.classifier = classifier
        self.width = width
        self.height = height
        self.complexity = complexity
        self.rotate = rotate
        self.running = False
        self.state = "waiting"
        self._cancel_requested = False

    @staticmethod
    def _stats(arr):
        """Return (min, mean, p50, p95, max) for a list of numbers in ms."""
        if not arr:
            return (0.0,) * 5
        a = np.asarray(arr)
        return (float(a.min()),
                float(a.mean()),
                float(np.percentile(a, 50)),
                float(np.percentile(a, 95)),
                float(a.max()))

    def run(self):
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=self.complexity,
        )

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)

        sequence_buffer = deque(maxlen=MAX_LEN)
        hands_lost = 0
        fps_history = deque(maxlen=30)        # for the smoothed UI FPS

        # ---- Performance tracking ----
        # Frame-level metrics, rolling window of last 200 frames
        mediapipe_ms_window = deque(maxlen=200)
        loop_ms_window      = deque(maxlen=200)
        # Inference latency, last 50 inferences
        inference_ms_window = deque(maxlen=50)

        # 5-second interval reporting
        report_interval_sec = 5.0
        report_start_time   = time.time()
        report_frame_count  = 0

        # Cumulative
        total_start = time.time()
        total_frames = 0

        self.running = True
        try:
            while self.running:
                t_loop_start = time.time()

                frame_bgr = picam2.capture_array()

                if self.rotate == 90:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotate == 180:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
                elif self.rotate == 270:
                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame_bgr = cv2.flip(frame_bgr, 1)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False

                # --- Time MediaPipe ---
                t_mp_start = time.time()
                results = holistic.process(frame_rgb)
                mp_ms = (time.time() - t_mp_start) * 1000.0
                mediapipe_ms_window.append(mp_ms)

                frame_rgb.flags.writeable = True

                if self._cancel_requested:
                    sequence_buffer.clear()
                    hands_lost = 0
                    self.state = "waiting"
                    self._cancel_requested = False
                    self.state_changed.emit("waiting")

                has_hand = hands_visible(results)

                if self.state == "waiting":
                    if has_hand:
                        sequence_buffer.clear()
                        sequence_buffer.append(extract_landmarks(results))
                        self.state = "recording"
                        hands_lost = 0
                        self.state_changed.emit("recording")

                elif self.state == "recording":
                    sequence_buffer.append(extract_landmarks(results))
                    if has_hand:
                        hands_lost = 0
                    else:
                        hands_lost = 1
                        self.state = "finishing"
                        self.state_changed.emit("finishing")

                elif self.state == "finishing":
                    if has_hand:
                        sequence_buffer.append(extract_landmarks(results))
                        hands_lost = 0
                        self.state = "recording"
                        self.state_changed.emit("recording")
                    else:
                        if hands_lost <= 5:
                            sequence_buffer.append(extract_landmarks(results))
                        hands_lost += 1

                        if hands_lost >= HANDS_LOST_FRAMES:
                            n = len(sequence_buffer)
                            if n >= MIN_FRAMES:
                                seq = np.stack(list(sequence_buffer), axis=0)

                                # --- Time TFLite inference ---
                                t_inf_start = time.time()
                                preds = self.classifier.predict(seq, top_k=3)
                                inf_ms = (time.time() - t_inf_start) * 1000.0
                                inference_ms_window.append(inf_ms)

                                print(f"\n[Inference] {n} frames, {inf_ms:.1f} ms")
                                for name, prob in preds:
                                    print(f"  {name:25s} {prob*100:5.1f}%")
                                # Show recent inference stats
                                imn, ime, ip50, ip95, imx = self._stats(
                                    list(inference_ms_window))
                                print(f"[Inference stats, last {len(inference_ms_window)}]: "
                                      f"min={imn:.1f}  mean={ime:.1f}  "
                                      f"p50={ip50:.1f}  p95={ip95:.1f}  max={imx:.1f} ms")
                                print()

                                self.result_ready.emit(preds, n)
                            else:
                                print(f"[Auto] discarded short ({n})")
                            sequence_buffer.clear()
                            hands_lost = 0
                            self.state = "waiting"
                            self.state_changed.emit("waiting")

                # Draw landmarks on preview
                mp_drawing.draw_landmarks(
                    frame_bgr, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    frame_bgr, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

                self.frame_ready.emit(frame_bgr)

                # Total loop time for this frame
                loop_ms = (time.time() - t_loop_start) * 1000.0
                loop_ms_window.append(loop_ms)

                dt = max(time.time() - t_loop_start, 1e-6)
                fps_history.append(1.0 / dt)
                self.fps_updated.emit(sum(fps_history) / len(fps_history))

                report_frame_count += 1
                total_frames += 1

                # ---- Periodic 5-second report ----
                now = time.time()
                if now - report_start_time >= report_interval_sec:
                    interval = now - report_start_time
                    avg_fps_5s = report_frame_count / interval

                    mp_min, mp_mean, mp_p50, mp_p95, mp_max = self._stats(
                        list(mediapipe_ms_window))
                    lp_min, lp_mean, lp_p50, lp_p95, lp_max = self._stats(
                        list(loop_ms_window))

                    total_elapsed = now - total_start
                    avg_fps_total = total_frames / max(total_elapsed, 1e-6)

                    print(f"[Perf @ {total_elapsed:6.1f}s] "
                          f"FPS_5s={avg_fps_5s:5.2f}  FPS_avg={avg_fps_total:5.2f}  "
                          f"frames={total_frames}")
                    print(f"  MediaPipe ms: min={mp_min:5.1f}  mean={mp_mean:5.1f}  "
                          f"p50={mp_p50:5.1f}  p95={mp_p95:5.1f}  max={mp_max:5.1f}")
                    print(f"  Loop      ms: min={lp_min:5.1f}  mean={lp_mean:5.1f}  "
                          f"p50={lp_p50:5.1f}  p95={lp_p95:5.1f}  max={lp_max:5.1f}")

                    report_start_time = now
                    report_frame_count = 0

        finally:
            picam2.stop()
            holistic.close()

            # ---- Final summary ----
            total_elapsed = time.time() - total_start
            avg_fps_total = total_frames / max(total_elapsed, 1e-6)
            print("\n" + "=" * 60)
            print("[Perf Final Summary]")
            print(f"  Total runtime: {total_elapsed:.1f} s")
            print(f"  Total frames:  {total_frames}")
            print(f"  Average FPS:   {avg_fps_total:.2f}")

            if mediapipe_ms_window:
                mp_stats = self._stats(list(mediapipe_ms_window))
                print(f"  MediaPipe ms (last {len(mediapipe_ms_window)}): "
                      f"min={mp_stats[0]:.1f}  mean={mp_stats[1]:.1f}  "
                      f"p50={mp_stats[2]:.1f}  p95={mp_stats[3]:.1f}  max={mp_stats[4]:.1f}")
            if loop_ms_window:
                lp_stats = self._stats(list(loop_ms_window))
                print(f"  Loop      ms (last {len(loop_ms_window)}): "
                      f"min={lp_stats[0]:.1f}  mean={lp_stats[1]:.1f}  "
                      f"p50={lp_stats[2]:.1f}  p95={lp_stats[3]:.1f}  max={lp_stats[4]:.1f}")
            if inference_ms_window:
                inf_stats = self._stats(list(inference_ms_window))
                print(f"  Inference ms (n={len(inference_ms_window)}): "
                      f"min={inf_stats[0]:.1f}  mean={inf_stats[1]:.1f}  "
                      f"p50={inf_stats[2]:.1f}  p95={inf_stats[3]:.1f}  max={inf_stats[4]:.1f}")
            print("=" * 60)

    def stop(self):
        self.running = False
        self.wait(2000)

    def request_cancel(self):
        self._cancel_requested = True


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, classifier, complexity, width, height, rotate, fullscreen):
        super().__init__()
        self.setWindowTitle("ASL")
        self.sentence_words = []

        self.banner_timer = QTimer(self)
        self.banner_timer.setSingleShot(True)
        self.banner_timer.timeout.connect(self._fade_result)

        self._build_ui()

        self.thread = InferenceThread(classifier, complexity, width, height, rotate)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.state_changed.connect(self._on_state)
        self.thread.fps_updated.connect(self._on_fps)
        self.thread.result_ready.connect(self._on_result)
        self.thread.start()

        if fullscreen:
            self.showFullScreen()
        else:
            self.resize(800, 480)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0d1117; color: #e6edf3; }
            QPushButton {
                background: #21262d; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { background: #30363d; }
        """)

        central = QWidget()
        self.setCentralWidget(central)

        # The big preview label - this is the centerpiece
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #000;")
        self.preview.setMinimumSize(640, 360)

        # Overlay labels - drawn on top of the preview
        # Status pill (top-left)
        self.status_label = QLabel("● WAITING", self.preview)
        self.status_label.setStyleSheet("""
            background: rgba(63, 185, 80, 200); color: white;
            border-radius: 14px; padding: 5px 14px;
            font-weight: 700; font-size: 13px;
        """)
        self.status_label.adjustSize()
        self.status_label.move(12, 12)

        # FPS (top-right) - position updated in resizeEvent
        self.fps_label = QLabel("FPS --", self.preview)
        self.fps_label.setStyleSheet("""
            background: rgba(0, 0, 0, 180); color: #c9d1d9;
            border-radius: 8px; padding: 4px 10px;
            font-family: monospace; font-size: 12px;
        """)

        # Big result banner (center, hidden until prediction comes in)
        self.result_banner = QLabel("", self.preview)
        self.result_banner.setAlignment(Qt.AlignCenter)
        self.result_banner.setStyleSheet("""
            background: rgba(0, 0, 0, 200); color: #ffffff;
            border-radius: 14px; padding: 14px 28px;
            font-weight: 800; font-size: 42px;
        """)
        self.result_banner.hide()

        # Layout: preview takes most of the space, sentence + buttons at bottom
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self.preview, stretch=1)

        # Sentence row
        self.sentence_label = QLabel("—")
        self.sentence_label.setStyleSheet("""
            background: #161b22; border: 1px solid #21262d; border-radius: 8px;
            padding: 8px 12px; font-size: 16px; color: #8b949e;
        """)
        self.sentence_label.setMinimumHeight(36)
        self.sentence_label.setMaximumHeight(50)
        root.addWidget(self.sentence_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(lambda: self.thread.request_cancel())

        self.btn_back = QPushButton("⌫")
        self.btn_back.clicked.connect(self._sentence_backspace)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._sentence_clear)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._sentence_copy)

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_quit)
        root.addLayout(btn_row)

    # ---- Resize: keep overlay labels positioned correctly ----
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlays()

    def _reposition_overlays(self):
        pw = self.preview.width()
        ph = self.preview.height()

        # FPS to top-right
        self.fps_label.adjustSize()
        self.fps_label.move(pw - self.fps_label.width() - 12, 12)

        # Result banner centered
        self.result_banner.adjustSize()
        bx = (pw - self.result_banner.width()) // 2
        by = (ph - self.result_banner.height()) // 2
        self.result_banner.move(bx, by)

    # ---- Slots ----
    def _on_frame(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        # Scale to fill the preview area while keeping aspect ratio
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _on_state(self, state):
        if state == "waiting":
            self.status_label.setText("● WAITING")
            self.status_label.setStyleSheet("""
                background: rgba(63, 185, 80, 200); color: white;
                border-radius: 14px; padding: 5px 14px;
                font-weight: 700; font-size: 13px;
            """)
        elif state == "recording":
            self.status_label.setText("● RECORDING")
            self.status_label.setStyleSheet("""
                background: rgba(248, 81, 73, 220); color: white;
                border-radius: 14px; padding: 5px 14px;
                font-weight: 700; font-size: 13px;
            """)
        elif state == "finishing":
            self.status_label.setText("● RELEASING")
            self.status_label.setStyleSheet("""
                background: rgba(240, 165, 0, 220); color: white;
                border-radius: 14px; padding: 5px 14px;
                font-weight: 700; font-size: 13px;
            """)
        self.status_label.adjustSize()

    def _on_fps(self, fps):
        self.fps_label.setText(f"FPS {fps:4.1f}")
        self._reposition_overlays()

    def _on_result(self, predictions, n_frames):
        if not predictions:
            return
        top_word, top_prob = predictions[0]

        # Show big banner over preview
        self.result_banner.setText(f"{top_word}   {top_prob*100:.0f}%")
        self.result_banner.adjustSize()
        self._reposition_overlays()
        self.result_banner.show()

        # Add to sentence
        if top_prob >= CONFIDENCE_THRESHOLD:
            self.sentence_words.append(top_word)
            self._refresh_sentence()

        self.banner_timer.start(2500)

    def _fade_result(self):
        self.result_banner.hide()

    # ---- Sentence ----
    def _refresh_sentence(self):
        if not self.sentence_words:
            self.sentence_label.setText("—")
            self.sentence_label.setStyleSheet("""
                background: #161b22; border: 1px solid #21262d; border-radius: 8px;
                padding: 8px 12px; font-size: 16px; color: #484f58;
            """)
        else:
            self.sentence_label.setText(" ".join(self.sentence_words))
            self.sentence_label.setStyleSheet("""
                background: #161b22; border: 1px solid #21262d; border-radius: 8px;
                padding: 8px 12px; font-size: 16px; color: #e6edf3;
            """)

    def _sentence_backspace(self):
        if self.sentence_words:
            self.sentence_words.pop()
            self._refresh_sentence()

    def _sentence_clear(self):
        self.sentence_words.clear()
        self._refresh_sentence()

    def _sentence_copy(self):
        text = " ".join(self.sentence_words)
        if text:
            QApplication.clipboard().setText(text)
            old = self.btn_copy.text()
            self.btn_copy.setText("✓")
            QTimer.singleShot(1000, lambda: self.btn_copy.setText(old))

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.tflite")
    parser.add_argument("--map", default="sign_to_prediction_index_map.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Rotate camera frame in degrees")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.map, args.threads)

    app = QApplication(sys.argv)
    win = MainWindow(classifier,
                     complexity=args.complexity,
                     width=args.width,
                     height=args.height,
                     rotate=args.rotate,
                     fullscreen=args.fullscreen)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()