"""
ASL Sign Language Recognition - Windows version (auto-trigger)
Based on Kaggle GISLR 1st place "Just Hands" solution.

Pipeline:
  Webcam -> MediaPipe Holistic -> sliding buffer -> TFLite model -> sign label

Trigger mode: AUTO
  - When at least one hand appears in frame -> start recording
  - When both hands are gone for ~1 second  -> stop & run inference
  - Result is displayed on screen for a few seconds
  - Repeat naturally for each sign

Hotkeys:
  - 'c' : cancel current recording
  - 'q' or ESC : quit
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import tensorflow as tf
Interpreter = tf.lite.Interpreter
print("[Info] Using tensorflow.lite.Interpreter")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8           # ignore recordings shorter than this
HANDS_LOST_FRAMES = 15   # how many "no hand" frames before triggering inference
                         # at ~15 FPS this is roughly 1 second
RESULT_DISPLAY_SEC = 4   # how long to keep showing the last result on screen

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
    def __init__(self, model_path, label_map_path, num_threads=4):
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

    def predict(self, sequence, top_k=3):
        if sequence.shape[0] > MAX_LEN:
            start = (sequence.shape[0] - MAX_LEN) // 2
            sequence = sequence[start:start + MAX_LEN]

        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'], list(sequence.shape)
        )
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(
            self.input_details[0]['index'], sequence.astype(np.float32)
        )
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details[0]['index']).flatten()

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(self.idx_to_sign[i], float(probs[i])) for i in top_idx]


# -----------------------------------------------------------------------------
# Landmark helpers
# -----------------------------------------------------------------------------
def extract_landmarks(results):
    """MediaPipe Holistic results -> (543, 3) array. Missing = NaN."""
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


def hands_visible(results):
    """True if MediaPipe detected at least one hand this frame."""
    return (results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_overlay(frame, *, state, frame_count, fps, predictions,
                 result_age, hands_lost):
    h, w = frame.shape[:2]

    if state == 'recording':
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        status = f"RECORDING  ({frame_count} frames)"
        color = (0, 0, 255)
    elif state == 'finishing':
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 4)
        remain = HANDS_LOST_FRAMES - hands_lost
        status = f"hold... releasing in {remain}"
        color = (0, 165, 255)
    else:
        status = "WAITING  -  show your hand to start"
        color = (0, 255, 0)
    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if predictions and result_age < RESULT_DISPLAY_SEC:
        top_name, top_prob = predictions[0]
        big_text = f"{top_name}  ({top_prob*100:.0f}%)"
        font_scale = 1.4
        thickness = 3
        (tw, th), _ = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        x = (w - tw) // 2
        y = h // 2
        cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20),
                      (0, 0, 0), -1)
        cv2.putText(frame, big_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness)

        for i, (name, prob) in enumerate(predictions[1:], start=1):
            t = f"{i+1}. {name}  {prob*100:.1f}%"
            cv2.putText(frame, t, (x, y + 40 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.putText(frame, "[C] cancel  [Q] quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.tflite')
    parser.add_argument('--map', default='sign_to_prediction_index_map.json')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.map, num_threads=args.threads)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.complexity,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Warn] DirectShow failed, trying Media Foundation backend")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {args.camera}.\n"
            "Make sure no other app (Zoom/Teams/browser) is using it.\n"
            "Try a different index: --camera 1"
        )

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] opened at {actual_w}x{actual_h}")

    # State machine:
    #   'waiting'   - no hand seen, idle
    #   'recording' - hand visible, accumulating frames
    #   'finishing' - hand was visible, now lost; counting down to inference
    state = 'waiting'
    sequence_buffer = deque(maxlen=MAX_LEN)
    hands_lost = 0

    last_predictions = []
    result_time = 0.0
    fps_history = deque(maxlen=30)

    print("\n" + "=" * 60)
    print("  ASL Real-time Inference (Windows, auto-trigger)")
    print("=" * 60)
    print("  Just show a sign with your hand(s), then drop them out of view.")
    print("  Result will appear automatically.")
    print("  C : cancel    Q/ESC : quit")
    print("=" * 60 + "\n")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[Camera] read failed")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            has_hand = hands_visible(results)

            # ------------------- State machine -------------------
            if state == 'waiting':
                if has_hand:
                    sequence_buffer.clear()
                    sequence_buffer.append(extract_landmarks(results))
                    state = 'recording'
                    hands_lost = 0
                    print("[Auto] hand detected, recording started")

            elif state == 'recording':
                sequence_buffer.append(extract_landmarks(results))
                if has_hand:
                    hands_lost = 0
                else:
                    hands_lost = 1
                    state = 'finishing'

            elif state == 'finishing':
                if has_hand:
                    sequence_buffer.append(extract_landmarks(results))
                    hands_lost = 0
                    state = 'recording'
                else:
                    if hands_lost <= 5:
                        sequence_buffer.append(extract_landmarks(results))
                    hands_lost += 1

                    if hands_lost >= HANDS_LOST_FRAMES:
                        n = len(sequence_buffer)
                        if n >= MIN_FRAMES:
                            seq = np.stack(list(sequence_buffer), axis=0)
                            t_inf = time.time()
                            last_predictions = classifier.predict(seq, top_k=3)
                            inf_ms = (time.time() - t_inf) * 1000
                            result_time = time.time()
                            print(f"\n[Inference] {n} frames in {inf_ms:.1f} ms")
                            for name, prob in last_predictions:
                                print(f"  {name:25s}  {prob*100:5.1f}%")
                            print()
                        else:
                            print(f"[Auto] discarded short recording ({n} frames)")
                        sequence_buffer.clear()
                        hands_lost = 0
                        state = 'waiting'

            # ------------------- Drawing -------------------
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

            result_age = time.time() - result_time if last_predictions else 999

            draw_overlay(frame, state=state,
                         frame_count=len(sequence_buffer),
                         fps=fps, predictions=last_predictions,
                         result_age=result_age, hands_lost=hands_lost)

            cv2.imshow('ASL Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if state != 'waiting':
                    sequence_buffer.clear()
                    hands_lost = 0
                    state = 'waiting'
                    print("[Auto] recording cancelled")
            elif key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == '__main__':
    main()