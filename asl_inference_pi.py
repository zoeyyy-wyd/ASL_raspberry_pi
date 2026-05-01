"""
ASL Sign Language Recognition - Raspberry Pi version
Camera Module 3 -> Picamera2 -> MediaPipe Holistic -> TFLite model -> sign label
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
    print("[Info] Using tflite_runtime.Interpreter")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[Info] Using tensorflow.lite.Interpreter")


ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8
HANDS_LOST_FRAMES = 15
RESULT_DISPLAY_SEC = 4

LH_START, LH_END = 468, 489
RH_START, RH_END = 522, 543


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
        print(f"[Model] Input dtype:   {self.input_details[0]['dtype']}")
        print(f"[Model] Output shape: {self.output_details[0]['shape']}")

        with open(label_map_path, encoding="utf-8") as f:
            sign_to_idx = json.load(f)

        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        print(f"[Model] Loaded {len(self.idx_to_sign)} classes")

    def predict(self, sequence, top_k=3):
        if sequence.shape[0] > MAX_LEN:
            start = (sequence.shape[0] - MAX_LEN) // 2
            sequence = sequence[start:start + MAX_LEN]

        self.interpreter.resize_tensor_input(
            self.input_details[0]["index"],
            list(sequence.shape)
        )
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            sequence.astype(np.float32)
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
            dtype=np.float32,
        )

    if results.left_hand_landmarks:
        out[LH_START:LH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )

    if results.pose_landmarks:
        out[489:522] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )

    if results.right_hand_landmarks:
        out[RH_START:RH_END] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )

    return out


def hands_visible(results):
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def draw_overlay(frame, state, frame_count, fps, predictions, result_age, hands_lost):
    h, w = frame.shape[:2]

    if state == "recording":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        status = f"RECORDING ({frame_count} frames)"
        color = (0, 0, 255)
    elif state == "finishing":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
        status = f"Releasing... {HANDS_LOST_FRAMES - hands_lost}"
        color = (0, 165, 255)
    else:
        status = "WAITING - show your hand"
        color = (0, 255, 0)

    cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if predictions and result_age < RESULT_DISPLAY_SEC:
        top_name, top_prob = predictions[0]
        text = f"{top_name} ({top_prob * 100:.0f}%)"

        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = max((w - tw) // 2, 10)
        y = h // 2

        cv2.rectangle(frame, (x - 15, y - th - 15), (x + tw + 15, y + 15), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        for i, (name, prob) in enumerate(predictions[1:], start=2):
            cv2.putText(
                frame,
                f"{i}. {name} {prob * 100:.1f}%",
                (x, y + 35 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )

    cv2.putText(frame, "[C] cancel  [Q] quit", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.tflite")
    parser.add_argument("--map", default="sign_to_prediction_index_map.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    classifier = SignClassifier(args.model, args.map, args.threads)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.complexity,
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    state = "waiting"
    sequence_buffer = deque(maxlen=MAX_LEN)
    hands_lost = 0
    last_predictions = []
    result_time = 0.0
    fps_history = deque(maxlen=30)

    print("=" * 60)
    print("ASL Real-time Inference on Raspberry Pi")
    print("Show your sign, then move hands out of frame.")
    print("C: cancel    Q/ESC: quit")
    print("=" * 60)

    try:
        while True:
            t0 = time.time()

            frame_rgb = picam2.capture_array()
            frame_rgb = cv2.flip(frame_rgb, 1)

            results = holistic.process(frame_rgb)
            has_hand = hands_visible(results)

            if state == "waiting":
                if has_hand:
                    sequence_buffer.clear()
                    sequence_buffer.append(extract_landmarks(results))
                    state = "recording"
                    hands_lost = 0
                    print("[Auto] hand detected, recording started")

            elif state == "recording":
                sequence_buffer.append(extract_landmarks(results))
                if has_hand:
                    hands_lost = 0
                else:
                    hands_lost = 1
                    state = "finishing"

            elif state == "finishing":
                if has_hand:
                    sequence_buffer.append(extract_landmarks(results))
                    hands_lost = 0
                    state = "recording"
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
                                print(f"  {name:25s} {prob * 100:5.1f}%")
                        else:
                            print(f"[Auto] discarded short recording ({n} frames)")

                        sequence_buffer.clear()
                        hands_lost = 0
                        state = "waiting"

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                frame_bgr,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            dt = max(time.time() - t0, 1e-6)
            fps_history.append(1.0 / dt)
            fps = sum(fps_history) / len(fps_history)

            result_age = time.time() - result_time if last_predictions else 999

            draw_overlay(
                frame_bgr,
                state=state,
                frame_count=len(sequence_buffer),
                fps=fps,
                predictions=last_predictions,
                result_age=result_age,
                hands_lost=hands_lost,
            )

            cv2.imshow("ASL Recognition - Raspberry Pi", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                sequence_buffer.clear()
                hands_lost = 0
                state = "waiting"
                print("[Auto] recording cancelled")
            elif key == ord("q") or key == 27:
                break

    finally:
        picam2.stop()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()