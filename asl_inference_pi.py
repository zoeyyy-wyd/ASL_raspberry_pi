"""
ASL Sign Language Recognition - Raspberry Pi version
Pipeline: Picamera2 -> MediaPipe Holistic -> sliding buffer -> TFLite -> sign label

Trigger mode: AUTO
  - Show your hand to start recording
  - Drop both hands out of frame for ~1s to trigger inference
  - Result is displayed for a few seconds, then it goes back to waiting

Hotkeys:
  - 'c'      : cancel current recording
  - 'q'/ESC  : quit

Setup (Raspberry Pi OS 64-bit, Bookworm):
  python3 -m venv asl --system-site-packages
  source asl/bin/activate
  pip install mediapipe opencv-python ai-edge-litert

  NOTE: We use ai-edge-litert (Google's renamed TFLite runtime) instead of
  the older tflite-runtime. The older tflite-runtime 2.14 on ARM has a bug
  where NaN values in the input cause logits to become NaN. ai-edge-litert
  handles NaN correctly, matching the reference implementation on x86.

Usage:
  python asl_inference_rpi.py
  python asl_inference_rpi.py --complexity 1   # higher MediaPipe accuracy
  python asl_inference_rpi.py --no-display     # headless mode (SSH without X)
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

# Use ai-edge-litert (Google's new name for tflite-runtime).
# This package correctly handles NaN inputs on ARM, unlike the older
# tflite-runtime which produced all-NaN outputs for the GISLR model.
from ai_edge_litert.interpreter import Interpreter
print("[Info] Using ai_edge_litert.Interpreter")


# ---------------------------------------------------------------------------yyuuh  --
# Constants - must match the training pipeline of the GISLR 1st place model
# -----------------------------------------------------------------------------
ROWS_PER_FRAME = 543
MAX_LEN = 384
MIN_FRAMES = 8           # ignore recordings shorter than this
HANDS_LOST_FRAMES = 15   # how many "no hand" frames before triggering inference
                         # at ~10-15 FPS on Pi this is roughly 1-1.5 seconds
RESULT_DISPLAY_SEC = 4   # how long to keep showing the last result on screen

# MediaPipe Holistic landmark layout:
#   face:        0..467     (468 points)
#   left_hand:   468..488   (21 points)
#   pose:        489..521   (33 points)
#   right_hand:  522..542   (21 points)
LH_START, LH_END = 468, 489
RH_START, RH_END = 522, 543


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

        print(f"[Model] Input shape:  {self.input_details[0]['shape']}")
        print(f"[Model] Input dtype:  {self.input_details[0]['dtype']}")
        print(f"[Model] Output shape: {self.output_details[0]['shape']}")

        with open(label_map_path, encoding="utf-8") as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        print(f"[Model] Loaded {len(self.idx_to_sign)} sign classes")

    def predict(self, sequence, top_k=3):
        """
        Args:
            sequence: (T, 543, 3) float32. NaN = landmark not detected.
        Returns:
            list of (sign_name, probability) sorted desc.
        """
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

        # Softmax for human-readable probabilities
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(self.idx_to_sign[int(i)], float(probs[i])) for i in top_idx]


# -----------------------------------------------------------------------------
# Landmark helpers
# -----------------------------------------------------------------------------
def extract_landmarks(results):
    """MediaPipe Holistic results -> (543, 3) array. Missing = NaN."""
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
    """True if MediaPipe detected at least one hand this frame."""
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_overlay(frame, *, state, frame_count, fps, predictions,
                 result_age, hands_lost):
    h, w = frame.shape[:2]

    if state == "recording":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        status = f"RECORDING ({frame_count} frames)"
        color = (0, 0, 255)
    elif state == "finishing":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 3)
        remain = HANDS_LOST_FRAMES - hands_lost
        status = f"Releasing... {remain}"
        color = (0, 165, 255)
    else:
        status = "WAITING - show your hand"
        color = (0, 255, 0)
    cv2.putText(frame, status, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if predictions and result_age < RESULT_DISPLAY_SEC:
        top_name, top_prob = predictions[0]
        text = f"{top_name} ({top_prob * 100:.0f}%)"

        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        x = max((w - tw) // 2, 10)
        y = h // 2

        cv2.rectangle(frame, (x - 15, y - th - 15),
                      (x + tw + 15, y + 15), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness)

        # Smaller list for top-2 and top-3
        for i, (name, prob) in enumerate(predictions[1:], start=2):
            cv2.putText(
                frame,
                f"{i}. {name} {prob * 100:.1f}%",
                (x, y + 25 + (i - 1) * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )

    cv2.putText(frame, "[C] cancel  [Q] quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.tflite")
    parser.add_argument("--map", default="sign_to_prediction_index_map.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threads", type=int, default=4,
                        help="TFLite CPU threads (Pi 5 has 4 cores)")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe model complexity (0=lite, 1=full, 2=heavy)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no GUI window). Useful for SSH.")
    args = parser.parse_args()

    # Initialize TFLite model
    classifier = SignClassifier(args.model, args.map, num_threads=args.threads)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.complexity,
    )

    # Initialize Pi Camera Module via Picamera2.
    # Note: Picamera2's "RGB888" format actually outputs pixels in BGR order
    # (legacy naming quirk), so we treat the captured array as BGR and
    # convert to RGB for MediaPipe.
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let auto-exposure settle
    print(f"[Camera] Picamera2 started at {args.width}x{args.height}")

    # State machine:
    #   'waiting'   - no hand seen, idle
    #   'recording' - hand visible, accumulating frames
    #   'finishing' - hand was visible, now lost; counting down to inference
    state = "waiting"
    sequence_buffer = deque(maxlen=MAX_LEN)
    hands_lost = 0
    last_predictions = []
    result_time = 0.0
    fps_history = deque(maxlen=30)

    print("\n" + "=" * 60)
    print("  ASL Real-time Inference (Raspberry Pi 5, auto-trigger)")
    print("=" * 60)
    print("  Show a sign with your hand(s), then drop them out of view.")
    print("  Result will appear automatically.")
    print("  C : cancel    Q/ESC : quit")
    print("=" * 60 + "\n")

    try:
        while True:
            t0 = time.time()

            # Picamera2 returns BGR despite the "RGB888" label
            frame_bgr = picam2.capture_array()
            frame_bgr = cv2.flip(frame_bgr, 1)  # mirror for natural use

            # Convert to true RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            has_hand = hands_visible(results)

            # ------------------- State machine -------------------
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
                    # Hand reappeared - keep recording
                    sequence_buffer.append(extract_landmarks(results))
                    hands_lost = 0
                    state = "recording"
                else:
                    # Still no hand - buffer a few "empty" frames so the model
                    # sees the natural end of the gesture
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
                            print(f"\n[Inference] {n} frames in {inf_ms:.1f} ms "
                                  f"(NaN ratio: {np.isnan(seq).mean():.3f})")
                            for name, prob in last_predictions:
                                print(f"  {name:25s}  {prob * 100:5.1f}%")
                            print()
                        else:
                            print(f"[Auto] discarded short recording ({n} frames)")
                        sequence_buffer.clear()
                        hands_lost = 0
                        state = "waiting"

            # ------------------- Drawing (only if display enabled) -------------------
            if not args.no_display:
                # Draw landmarks directly on the BGR frame
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
                    if state != "waiting":
                        sequence_buffer.clear()
                        hands_lost = 0
                        state = "waiting"
                        print("[Auto] recording cancelled")
                elif key == ord("q") or key == 27:  # 27 = ESC
                    break
            else:
                # Headless mode: just track FPS in console
                dt = max(time.time() - t0, 1e-6)
                fps_history.append(1.0 / dt)

    finally:
        picam2.stop()
        holistic.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("\n[Exit] Cleaned up resources.")


if __name__ == "__main__":
    main()