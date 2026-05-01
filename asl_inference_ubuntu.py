"""
ASL Sign Language Recognition - Local PC inference (Ubuntu + TensorFlow)
Based on Kaggle GISLR 1st place "Just Hands" solution.

Pipeline:
  Webcam -> MediaPipe Holistic -> buffer -> TFLite model -> sign label

Trigger mode: SPACEBAR
  - Press SPACE to start recording a sign
  - Press SPACE again to stop and run inference
  - Press 'c' to cancel current recording
  - Press 'q' or ESC to quit

Setup (Ubuntu):
  python3 -m venv asl_env
  source asl_env/bin/activate
  pip install --upgrade pip
  pip install mediapipe opencv-python numpy tensorflow

Usage:
  python asl_inference_pc.py
  # or with explicit paths:
  python asl_inference_pc.py --model model.tflite --map sign_to_prediction_index_map.json
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


try:
    from tflite_runtime.interpreter import Interpreter
    print("[Info] Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[Info] Using tensorflow.lite")

# -----------------------------------------------------------------------------
# Constants — must match the training pipeline exactly
# -----------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8

# MediaPipe Holistic landmark layout:
#   face:        0..467     (468 points)
#   left_hand:   468..488   (21 points)
#   pose:        489..521   (33 points)
#   right_hand:  522..542   (21 points)
LH_START, LH_END = 468, 489
RH_START, RH_END = 522, 543


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class SignClassifier:
    def __init__(self, model_path: str, label_map_path: str, num_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"[Model] Input shape:  {self.input_details[0]['shape']}")
        print(f"[Model] Input dtype:  {self.input_details[0]['dtype']}")
        print(f"[Model] Output shape: {self.output_details[0]['shape']}")

        with open(label_map_path, encoding='utf-8') as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        print(f"[Model] Loaded {len(self.idx_to_sign)} sign classes")

    def predict(self, sequence: np.ndarray, top_k: int = 3):
        """
        Args:
            sequence: (T, 543, 3) float32. NaN = landmark not detected.
        Returns:
            list of (sign_name, probability) sorted by prob desc.
        """
        if sequence.shape[0] > MAX_LEN:
            start = (sequence.shape[0] - MAX_LEN) // 2
            sequence = sequence[start:start + MAX_LEN]

        # Resize for dynamic frame dim
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'], list(sequence.shape)
        )
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(
            self.input_details[0]['index'], sequence.astype(np.float32)
        )
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details[0]['index']).flatten()

        # Softmax for readable probabilities
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(self.idx_to_sign[i], float(probs[i])) for i in top_idx]


# -----------------------------------------------------------------------------
# Landmark extraction
# -----------------------------------------------------------------------------
def extract_landmarks(results) -> np.ndarray:
    """
    MediaPipe Holistic results -> (543, 3) array.
    Missing landmarks become NaN (model expects this).
    """
    out = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

    if results.face_landmarks:
        out[0:468] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
            dtype=np.float32
        )
    if results.left_hand_landmarks:
        out[LH_START:LH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )
    if results.pose_landmarks:
        out[489:522] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
    if results.right_hand_landmarks:
        out[RH_START:RH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )
    return out


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def draw_overlay(frame, *, recording: bool, frame_count: int,
                 fps: float, predictions: list):
    h, w = frame.shape[:2]

    if recording:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        status = f"RECORDING  frames: {frame_count}"
        color = (0, 0, 255)
    else:
        status = "READY  -  press SPACE to record"
        color = (0, 255, 0)
    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if predictions:
        cv2.rectangle(frame, (10, 75), (350, 75 + 35 * len(predictions) + 10),
                      (40, 40, 40), -1)
        for i, (name, prob) in enumerate(predictions):
            text = f"{i+1}. {name}  {prob*100:.1f}%"
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, text, (15, 105 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.putText(frame, "[SPACE] start/stop  [C] cancel  [Q] quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.tflite',
                        help='Path to TFLite model')
    parser.add_argument('--map', default='sign_to_prediction_index_map.json',
                        help='Path to sign_to_prediction_index_map.json')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default 0, usually /dev/video0)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threads', type=int, default=4,
                        help='TFLite CPU threads')
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2],
                        help='MediaPipe model complexity (0=lite, 1=full, 2=heavy)')
    args = parser.parse_args()

    # Init model
    classifier = SignClassifier(args.model, args.map, num_threads=args.threads)

    # Init MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.complexity,
    )

    # Open webcam — on Linux, V4L2 is the standard backend
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        print("[Warn] V4L2 backend failed, trying default")
        cap = cv2.VideoCapture(args.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # MJPG often gives higher FPS than YUYV on Linux webcams
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {args.camera}.\n"
            f"Check: ls /dev/video*\n"
            f"And:   sudo usermod -aG video $USER  (then re-login)"
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] opened at {actual_w}x{actual_h}")

    # State
    sequence_buffer = deque(maxlen=MAX_LEN)
    recording = False
    last_predictions = []
    fps_history = deque(maxlen=30)

    print("\n" + "=" * 60)
    print("  ASL Real-time Inference")
    print("=" * 60)
    print("  SPACE : start / stop recording")
    print("  C     : cancel current recording")
    print("  Q/ESC : quit")
    print("=" * 60 + "\n")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[Camera] read failed")
                break

            frame = cv2.flip(frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            if recording:
                landmarks = extract_landmarks(results)
                sequence_buffer.append(landmarks)

            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())

            dt = max(time.time() - t0, 1e-6)
            fps_history.append(1.0 / dt)
            fps = sum(fps_history) / len(fps_history)

            draw_overlay(frame, recording=recording,
                         frame_count=len(sequence_buffer),
                         fps=fps, predictions=last_predictions)

            cv2.imshow('ASL Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not recording:
                    sequence_buffer.clear()
                    recording = True
                    print("[Record] started")
                else:
                    recording = False
                    n = len(sequence_buffer)
                    if n < MIN_FRAMES:
                        print(f"[Record] too short ({n} frames), need >= {MIN_FRAMES}")
                        last_predictions = []
                    else:
                        seq = np.stack(list(sequence_buffer), axis=0)
                        t_inf = time.time()
                        last_predictions = classifier.predict(seq, top_k=3)
                        inf_ms = (time.time() - t_inf) * 1000
                        print(f"\n[Inference] {n} frames in {inf_ms:.1f} ms")
                        for name, prob in last_predictions:
                            print(f"  {name:25s}  {prob*100:5.1f}%")
                        print()
            elif key == ord('c'):
                if recording:
                    sequence_buffer.clear()
                    recording = False
                    print("[Record] cancelled")
            elif key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == '__main__':
    main()
