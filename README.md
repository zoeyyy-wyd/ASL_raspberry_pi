# Real-Time ASL Translation on Raspberry Pi

A fully offline, edge-deployed American Sign Language (ASL) recognition system running on a Raspberry Pi 5. The system recognizes 250 common ASL signs in real time using a camera feed, with no internet connection required.

**EECS E6765 Embedded AI — Spring 2026**
Yidan Wang (yw4636) · Hanling Yao (hy2944) · Columbia University

---

## Demo

> Raise your hand → sign a word → lower your hand → word appears on screen

The system accumulates recognized words into a sentence on a touchscreen display, supporting word-level deletion and full sentence reset.

---

## How It Works

```
Camera (30 fps)
    ↓
MediaPipe Holistic  →  543 landmarks/frame
    ↓
Frame Buffer  →  triggered by hand raise / lower
    ↓
TFLite Inference  →  Conv1D + Transformer, 3.6 MB
    ↓
Touchscreen Display  →  sentence accumulation
```

1. **MediaPipe Holistic** extracts 543 landmarks per frame (face, pose, hands) from the live camera feed
2. When a hand is detected, the system enters **recording** state and buffers landmark frames
3. When the hand leaves frame for ~1 second, the buffer is passed to the **TFLite model** for inference
4. The predicted sign is appended to the sentence on the **touchscreen frontend**

All processing runs locally on the Raspberry Pi 5 — no cloud, no network dependency.

---

## Model

- **Architecture**: Stacked Conv1D blocks (kernel size 17) + Transformer blocks with multi-head self-attention, repeated twice
- **Input**: 42 hand landmarks per frame (left + right hand), with first- and second-order temporal differences (velocity + acceleration)
- **Output**: 250-class classification over common ASL signs
- **Training data**: [Google Isolated Sign Language Recognition (GISLR) dataset](https://www.kaggle.com/competitions/asl-signs) — 94,000 clips, 21 native signers
- **Quantization**: FP16 post-training quantization via TFLite
- **Model size**: 3.6 MB
- **Accuracy**: ~70% top-1 on GISLR validation set

---

## Hardware

| Component | Spec |
|-----------|------|
| Compute | Raspberry Pi 5 (8 GB RAM) |
| Camera | Raspberry Pi Camera Module 3 |
| Display | 5" Touchscreen Monitor |
| Storage | 32 GB microSD (Class 10) |

---

## Performance (Raspberry Pi 5)

| Metric | Value |
|--------|-------|
| Average FPS | 17–18 |
| Peak FPS | 24 |
| MediaPipe latency (mean) | 37–61 ms/frame |
| TFLite inference (mean) | 5.6 ms/gesture |
| TFLite inference (p95) | 7.5 ms/gesture |
| Model size | 3.6 MB |
| Top-1 accuracy | ~70% |

---

## Installation

### Requirements

- Raspberry Pi 5 (8 GB recommended)
- Raspberry Pi OS (64-bit)
- Python 3.11+
- Raspberry Pi Camera Module 3
- 5" touchscreen display

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/asl-translation-pi.git
cd asl-translation-pi

# Install dependencies
pip install opencv-python picamera2 mediapipe ai-edge-litert numpy
```

> **Important**: Use `ai-edge-litert` instead of `tflite-runtime`. The older `tflite-runtime 2.14` has a bug on ARM that causes NaN propagation through the model, resulting in complete inference failure. See [Deployment Notes](#deployment-notes) below.

### Run

```bash
python asl_inference_pi.py
```

---

## Touchscreen Frontend

The frontend displays recognized words as a growing sentence and provides four controls:

| Button | Action |
|--------|--------|
| **Cancel** | Discard current gesture without running inference |
| **Backspace** | Remove the last recognized word |
| **Clear** | Reset the entire sentence |
| **Copy** | Copy the sentence to clipboard |

The top-left corner shows the current system state (`WAITING` / `RECORDING` / `FINISHING`) and the top-right shows live FPS.

---

## Deployment Notes

### Critical: NaN Inference Bug on ARM

When deploying on Raspberry Pi, **do not use `tflite-runtime`**. Version 2.14 has a bug on ARM where `tf.where(is_nan(x), 0, x)` operations do not correctly eliminate NaN values, causing them to propagate through the entire network and producing all-NaN outputs.

**Fix**: Replace with `ai-edge-litert`, Google's updated successor package:

```bash
pip uninstall tflite-runtime -y
pip install ai-edge-litert
```

Change the import in your code:

```python
# Before
from tflite_runtime.interpreter import Interpreter

# After
from ai_edge_litert.interpreter import Interpreter
```

### MediaPipe Version

Ensure MediaPipe versions match between your development machine and the Pi. Version mismatches can cause subtle landmark format differences that degrade inference quality.

---

## Project Structure

```
asl-translation-pi/
├── asl_inference_pi.py     # Main inference script
├── model.tflite            # Trained TFLite model (3.6 MB)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Training

The model was trained from scratch on the [GISLR dataset](https://www.kaggle.com/competitions/asl-signs) using TensorFlow with:
- **Optimizer**: Adam with cosine learning rate schedule
- **Augmentation**: Random temporal cropping, landmark coordinate jitter
- **Quantization**: FP16 post-training quantization via TFLite converter

Training was performed on a GPU; only the exported `.tflite` file is needed for inference on the Pi.

---

## Recognized Signs

The model supports 250 ASL signs from the GISLR dataset, covering common conversational vocabulary including words like `hello`, `thankyou`, `mom`, `dad`, `drink`, `eat`, `help`, `happy`, `animal`, and many more.

Signs with distinctive static hand shapes tend to be recognized with higher confidence. Signs that rely primarily on hand motion trajectory are more challenging and may require clearer, more deliberate signing.

**Note**: The model performs better on right-hand dominant signing, reflecting the composition of the GISLR training set.

---

## References

- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [Google Isolated Sign Language Recognition — Kaggle](https://www.kaggle.com/competitions/asl-signs)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [ai-edge-litert](https://pypi.org/project/ai-edge-litert/)

---

## License

MIT License
