"""
ASL Sign Language Recognition - Raspberry Pi PyQt5 Desktop GUI

Layout:
  +-------------------------------------------------+
  | [STATUS PILL]                          [FPS]    |
  |                                                 |
  |              ┌───────────────┐                  |
  |              │    HELLO      │  <- big result   |
  |              │    87%        │                  |
  |              └───────────────┘                  |
  |    bye 8%   hi 3%                               |
  |                                                 |
  |    ┌── camera preview (small) ───┐              |
  |    │                             │              |
  |    └─────────────────────────────┘              |
  |                                                 |
  |   SENTENCE                                      |
  |   ┌──────────────────────────────────────────┐  |
  |   │ hello mom drink water                    │  |
  |   └──────────────────────────────────────────┘  |
  |   [Cancel] [Backspace] [Clear] [Copy]           |
  +-------------------------------------------------+

Setup (Raspberry Pi):
  source myenv/bin/activate
  sudo apt install python3-pyqt5

Usage:
  python asl_qt_gui.py
  python asl_qt_gui.py --fullscreen   # for the Pi touchscreen
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
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame
)

try:
    from tflite_runtime.interpreter import Interpreter
    print("[Info] Using tflite_runtime.Interpreter")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[Info] Using tensorflow.lite.Interpreter")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8
HANDS_LOST_FRAMES = 15
LH_START, LH_END = 468, 489
RH_START, RH_END = 522, 543

CONFIDENCE_THRESHOLD = 0.30   # min prob to add a word to the sentence


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------
class SignClassifier:
    def __init__(self, model_path, label_map_path, num_threads=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

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
            self.input_details[0]["index"], list(sequence.shape)
        )
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(
            self.input_details[0]["index"], sequence.astype(np.float32)
        )
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        ).flatten()

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
# Worker thread - capture + MediaPipe + inference
# Emits Qt signals to update the GUI on the main thread.
# -----------------------------------------------------------------------------
class InferenceThread(QThread):
    frame_ready    = pyqtSignal(np.ndarray)              # latest BGR frame
    state_changed  = pyqtSignal(str)                     # waiting/recording/finishing
    fps_updated    = pyqtSignal(float, int)              # fps, buffer_size
    result_ready   = pyqtSignal(list, int, float)        # predictions, n_frames, ms

    def __init__(self, classifier, complexity=1, width=640, height=480):
        super().__init__()
        self.classifier = classifier
        self.width = width
        self.height = height
        self.complexity = complexity
        self.running = False
        self.state = "waiting"
        self._cancel_requested = False

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
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)

        sequence_buffer = deque(maxlen=MAX_LEN)
        hands_lost = 0
        fps_history = deque(maxlen=30)

        self.running = True
        try:
            while self.running:
                t0 = time.time()

                frame_bgr = picam2.capture_array()
                frame_bgr = cv2.flip(frame_bgr, 1)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True

                # Handle cancel from UI thread
                if self._cancel_requested:
                    sequence_buffer.clear()
                    hands_lost = 0
                    self.state = "waiting"
                    self._cancel_requested = False
                    self.state_changed.emit("waiting")

                has_hand = hands_visible(results)

                # ---- State machine ----
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
                                t_inf = time.time()
                                preds = self.classifier.predict(seq, top_k=3)
                                inf_ms = (time.time() - t_inf) * 1000

                                print(f"[Inference] {n} frames, {inf_ms:.1f} ms")
                                for name, prob in preds:
                                    print(f"  {name:25s} {prob*100:5.1f}%")

                                self.result_ready.emit(preds, n, inf_ms)
                            else:
                                print(f"[Auto] discarded short ({n} frames)")

                            sequence_buffer.clear()
                            hands_lost = 0
                            self.state = "waiting"
                            self.state_changed.emit("waiting")

                # Draw landmarks on preview frame
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

                dt = max(time.time() - t0, 1e-6)
                fps_history.append(1.0 / dt)
                fps = sum(fps_history) / len(fps_history)
                self.fps_updated.emit(fps, len(sequence_buffer))

        finally:
            picam2.stop()
            holistic.close()

    def stop(self):
        self.running = False
        self.wait(2000)

    def request_cancel(self):
        self._cancel_requested = True


# -----------------------------------------------------------------------------
# Main GUI window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, classifier, complexity, width, height, fullscreen):
        super().__init__()
        self.setWindowTitle("ASL Sign Recognition")
        self._apply_stylesheet()

        self.sentence_words = []   # accumulated recognized words

        # Banner timer - hides the result after a few seconds
        self.banner_timer = QTimer(self)
        self.banner_timer.setSingleShot(True)
        self.banner_timer.timeout.connect(self._fade_result)

        # Build UI
        self._build_ui()

        # Worker thread
        self.thread = InferenceThread(classifier, complexity, width, height)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.state_changed.connect(self._on_state)
        self.thread.fps_updated.connect(self._on_fps)
        self.thread.result_ready.connect(self._on_result)
        self.thread.start()

        if fullscreen:
            self.showFullScreen()
        else:
            self.resize(900, 760)

    # ------------- Style -------------
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0d1117; color: #e6edf3; }

            QLabel#StatusPill {
                background: rgba(63, 185, 80, 50);
                color: #3fb950;
                border: 2px solid #3fb950;
                border-radius: 18px;
                padding: 6px 18px;
                font-weight: 700;
                font-size: 13px;
            }
            QLabel#StatusPill[state="recording"] {
                background: rgba(248, 81, 73, 60);
                color: #f85149;
                border-color: #f85149;
            }
            QLabel#StatusPill[state="finishing"] {
                background: rgba(240, 165, 0, 60);
                color: #f0a500;
                border-color: #f0a500;
            }

            QLabel#FpsLabel {
                background: #161b22;
                color: #8b949e;
                border-radius: 8px;
                padding: 5px 14px;
                font-family: monospace;
                font-size: 12px;
            }

            QFrame#CenterCard {
                background: #161b22;
                border-radius: 16px;
                border: 1px solid #21262d;
            }
            QFrame#CenterCard[active="true"] {
                border-color: #3fb950;
            }

            QLabel#CenterWord {
                color: #ffffff;
                font-weight: 800;
                font-size: 80px;
                letter-spacing: -2px;
            }
            QLabel#CenterWordIdle {
                color: #30363d;
                font-weight: 800;
                font-size: 80px;
                letter-spacing: -2px;
            }

            QLabel#CenterProb {
                background: rgba(63, 185, 80, 50);
                color: #3fb950;
                border: 1px solid #3fb950;
                border-radius: 14px;
                padding: 4px 16px;
                font-weight: 600;
                font-size: 16px;
            }

            QLabel#Runner {
                background: #161b22;
                color: #8b949e;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
            }

            QFrame#PreviewFrame {
                background: #161b22;
                border-radius: 12px;
                border: 1px solid #21262d;
            }

            QLabel#SectionHeader {
                color: #6e7681;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
            }

            QLabel#SentenceText {
                color: #e6edf3;
                background: #161b22;
                border-radius: 10px;
                border: 1px solid #21262d;
                padding: 14px 16px;
                font-size: 18px;
                min-height: 36px;
            }

            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover  { background: #30363d; border-color: #484f58; }
            QPushButton:pressed { background: #161b22; }
        """)

    # ------------- UI construction -------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ---- Top bar: status pill (left) + FPS (right) ----
        top_bar = QHBoxLayout()
        self.status_pill = QLabel("● WAITING")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setProperty("state", "waiting")

        self.fps_label = QLabel("FPS --   0 frames")
        self.fps_label.setObjectName("FpsLabel")

        top_bar.addWidget(self.status_pill)
        top_bar.addStretch()
        top_bar.addWidget(self.fps_label)
        root.addLayout(top_bar)

        # ---- Center: big word + prob + runners-up ----
        self.center_card = QFrame()
        self.center_card.setObjectName("CenterCard")
        self.center_card.setProperty("active", "false")
        center_layout = QVBoxLayout(self.center_card)
        center_layout.setContentsMargins(20, 24, 20, 24)
        center_layout.setSpacing(12)
        center_layout.setAlignment(Qt.AlignCenter)

        self.center_word = QLabel("—")
        self.center_word.setObjectName("CenterWordIdle")
        self.center_word.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.center_word)

        self.center_prob = QLabel("waiting for sign...")
        self.center_prob.setObjectName("CenterProb")
        self.center_prob.setAlignment(Qt.AlignCenter)
        self.center_prob.setStyleSheet(
            "background: #161b22; color: #6e7681; border: 1px solid #21262d;"
        )
        # Make it shrink-wrap to text
        self.center_prob.setSizePolicy(self.center_prob.sizePolicy().Maximum,
                                       self.center_prob.sizePolicy().Fixed)
        prob_wrap = QHBoxLayout()
        prob_wrap.addStretch()
        prob_wrap.addWidget(self.center_prob)
        prob_wrap.addStretch()
        center_layout.addLayout(prob_wrap)

        # Runners (top-2, top-3)
        runners_wrap = QHBoxLayout()
        runners_wrap.setSpacing(8)
        runners_wrap.addStretch()
        self.runner2 = QLabel("")
        self.runner3 = QLabel("")
        self.runner2.setObjectName("Runner")
        self.runner3.setObjectName("Runner")
        self.runner2.hide()
        self.runner3.hide()
        runners_wrap.addWidget(self.runner2)
        runners_wrap.addWidget(self.runner3)
        runners_wrap.addStretch()
        center_layout.addLayout(runners_wrap)

        root.addWidget(self.center_card, stretch=2)

        # ---- Camera preview (smaller) ----
        preview_header = QLabel("CAMERA")
        preview_header.setObjectName("SectionHeader")
        root.addWidget(preview_header)

        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("PreviewFrame")
        prev_layout = QVBoxLayout(self.preview_frame)
        prev_layout.setContentsMargins(8, 8, 8, 8)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setMaximumHeight(280)
        prev_layout.addWidget(self.preview_label)

        root.addWidget(self.preview_frame, stretch=2)

        # ---- Sentence builder ----
        sentence_header = QLabel("SENTENCE")
        sentence_header.setObjectName("SectionHeader")
        root.addWidget(sentence_header)

        self.sentence_label = QLabel("")
        self.sentence_label.setObjectName("SentenceText")
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setText(
            "<i style='color:#484f58;'>Recognized words will appear here...</i>"
        )
        root.addWidget(self.sentence_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_cancel = QPushButton("Cancel recording")
        self.btn_cancel.clicked.connect(self._cancel_recording)

        self.btn_back = QPushButton("⌫ Backspace")
        self.btn_back.clicked.connect(self._sentence_backspace)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._sentence_clear)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._sentence_copy)

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_quit)
        root.addLayout(btn_row)

    # ------------- Slots: receive signals from worker -------------
    def _on_frame(self, frame_bgr):
        """Update the preview image."""
        h, w, _ = frame_bgr.shape
        # Convert BGR -> RGB for QImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)

    def _on_state(self, state):
        """Update the status pill."""
        self.status_pill.setProperty("state", state)
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

        labels = {
            "waiting":   "● WAITING",
            "recording": "● RECORDING",
            "finishing": "● RELEASING",
        }
        self.status_pill.setText(labels.get(state, state.upper()))

    def _on_fps(self, fps, buffer_size):
        self.fps_label.setText(f"FPS {fps:4.1f}   {buffer_size:3d} frames")

    def _on_result(self, predictions, n_frames, ms):
        """Show the big result and add to sentence."""
        if not predictions:
            return
        top_word, top_prob = predictions[0]

        # Big center display
        self.center_word.setObjectName("CenterWord")
        self.center_word.setStyleSheet("color: #ffffff;")  # ensure refresh
        self.center_word.setText(top_word)

        self.center_prob.setStyleSheet(
            "background: rgba(63, 185, 80, 50); color: #3fb950; "
            "border: 1px solid #3fb950; border-radius: 14px; "
            "padding: 4px 16px; font-weight: 600; font-size: 16px;"
        )
        self.center_prob.setText(f"{top_prob*100:.0f}%   ·   {n_frames} frames")

        # Runners-up
        if len(predictions) > 1:
            r2_word, r2_prob = predictions[1]
            self.runner2.setText(f"{r2_word}  {r2_prob*100:.0f}%")
            self.runner2.show()
        else:
            self.runner2.hide()

        if len(predictions) > 2:
            r3_word, r3_prob = predictions[2]
            self.runner3.setText(f"{r3_word}  {r3_prob*100:.0f}%")
            self.runner3.show()
        else:
            self.runner3.hide()

        # Highlight center card
        self.center_card.setProperty("active", "true")
        self.center_card.style().unpolish(self.center_card)
        self.center_card.style().polish(self.center_card)

        # Add to sentence if confidence is high enough
        if top_prob >= CONFIDENCE_THRESHOLD:
            self.sentence_words.append(top_word)
            self._refresh_sentence()

        # Schedule fade-out after 4 seconds
        self.banner_timer.start(4000)

    def _fade_result(self):
        """Reset center to idle look."""
        self.center_word.setObjectName("CenterWordIdle")
        self.center_word.setStyleSheet("color: #30363d;")
        self.center_word.setText("—")
        self.center_prob.setStyleSheet(
            "background: #161b22; color: #6e7681; "
            "border: 1px solid #21262d; border-radius: 14px; "
            "padding: 4px 16px; font-size: 14px;"
        )
        self.center_prob.setText("waiting for sign...")
        self.runner2.hide()
        self.runner3.hide()

        self.center_card.setProperty("active", "false")
        self.center_card.style().unpolish(self.center_card)
        self.center_card.style().polish(self.center_card)

    def reset_center(self):
        self._fade_result()

    # ------------- Sentence operations -------------
    def _refresh_sentence(self):
        if not self.sentence_words:
            self.sentence_label.setText(
                "<i style='color:#484f58;'>Recognized words will appear here...</i>"
            )
            return
        # Render words as "chips" with HTML (Qt label supports basic HTML)
        chips = []
        for w in self.sentence_words:
            chips.append(
                f'<span style="background:#21262d; color:#e6edf3; '
                f'padding:3px 10px; border-radius:6px; margin:2px;">'
                f'{w}</span>'
            )
        self.sentence_label.setText(" ".join(chips))

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
            # Brief visual hint
            old = self.btn_copy.text()
            self.btn_copy.setText("✓ Copied")
            QTimer.singleShot(1200, lambda: self.btn_copy.setText(old))

    def _cancel_recording(self):
        self.thread.request_cancel()

    # ------------- Cleanup -------------
    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.tflite")
    parser.add_argument("--map", default="sign_to_prediction_index_map.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.map, args.threads)

    app = QApplication(sys.argv)
    win = MainWindow(
        classifier,
        complexity=args.complexity,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
    )
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
