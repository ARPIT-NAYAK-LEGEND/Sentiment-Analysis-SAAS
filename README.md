# Multimodal Sentiment & Emotion Analysis (MELD)

This project implements a **multimodal deep learning model** for **emotion and sentiment classification** using **text, audio, and video** modalities.  
The model is trained on the **MELD (Multimodal EmotionLines Dataset)** and deployed using **AWS SageMaker** with GPU acceleration.

---

## 🚀 Key Features

- **Text modality**: BERT-based frozen text encoder
- **Video modality**: 3D ResNet (R3D-18) for spatiotemporal feature extraction
- **Audio modality**: Mel-spectrogram based CNN encoder
- **Multimodal fusion**: Late fusion with joint emotion & sentiment heads
- **Emotion classification**: 7 emotion classes
- **Sentiment classification**: 3 sentiment classes
- **Training infrastructure**: AWS SageMaker (GPU – ml.g5.xlarge)
- **Logging**: TensorBoard + CloudWatch
- **Checkpointing**: Best model saved automatically

---

## 🧠 Model Architecture

- **Text Encoder**: `bert-base-uncased` (frozen)
- **Video Encoder**: R3D-18 (pretrained on Kinetics)
- **Audio Encoder**: CNN over Mel-spectrograms
- **Fusion Layer**: Concatenation → FC → BatchNorm → ReLU
- **Outputs**:
  - Emotion classifier (7 classes)
  - Sentiment classifier (3 classes)

---

## 📊 Dataset

- **Dataset**: MELD (Multimodal EmotionLines Dataset)
- **Inputs**:
  - Utterance text
  - Corresponding video clips
  - Audio extracted from video using FFmpeg
- **Labels**:
  - Emotion (anger, disgust, fear, joy, neutral, sadness, surprise)
  - Sentiment (negative, neutral, positive)

---

## ⚙️ Training Details

- **Framework**: PyTorch
- **Optimizer**: AdamW
- **Loss**: Weighted Cross-Entropy with label smoothing
- **Batch size**: 32
- **Epochs**: Configurable
- **Device**: NVIDIA A10G GPU
- **Runtime control**: SageMaker `max_run`

---

## ☁️ AWS SageMaker Integration

- Custom PyTorch training container
- GPU-backed training jobs
- Automatic model artifact generation (`model.tar.gz`)
- TensorBoard logs stored in Amazon S3
- Supports long-running training with configurable runtime limits

---

## 📦 Project Structure

