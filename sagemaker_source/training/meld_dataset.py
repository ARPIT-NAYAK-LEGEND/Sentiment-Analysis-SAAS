from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
from scipy.io import wavfile
import tempfile
import imageio_ffmpeg


class MeldDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)

        def _norm(col: str) -> str:
            return col.strip().lower().replace(" ", "_").replace(".", "").replace("-", "_")

        self.data.columns = [_norm(c) for c in self.data.columns]

        self.video_dir = str(video_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sad": 5,
            "sadness": 5,
            "surprise": 6,
        }

        self.sentiment_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

    # =========================
    # VIDEO
    # =========================
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        is_valid = True

        try:
            if not cap.isOpened():
                is_valid = False
            else:
                while len(frames) < 30:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            # fallback black frames
            frames = [np.zeros((224, 224, 3), dtype=np.float32)] * 30
            is_valid = False

        while len(frames) < 30:
            frames.append(frames[-1])

        frames = np.stack(frames[:30])
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T,C,H,W)

        return frames, is_valid

    # =========================
    # AUDIO
    # =========================
    def _extract_audio_features(self, video_path):
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    audio_path,
                    "-y",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )

            sr, waveform = wavfile.read(audio_path)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64
            )(waveform)

            mel = torchaudio.transforms.AmplitudeToDB()(mel)
            return mel

        except Exception:
            return torch.zeros(1, 64, 100)

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # =========================
    # DATASET
    # =========================
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[int(idx)]

        video_name = f"dia{row['dialogue_id']}_utt{row['utterance_id']}.mp4"
        video_path = os.path.join(self.video_dir, video_name)

        if not os.path.exists(video_path):
            return None

        text = self.tokenizer(
            row["utterance"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        video, is_valid = self._load_video_frames(video_path)

        if is_valid:
            audio = self._extract_audio_features(video_path)
        else:
            audio = torch.zeros(1, 64, 100)

        emotion = self.emotion_map.get(row["emotion"].lower(), 4)
        sentiment = self.sentiment_map.get(row["sentiment"].lower(), 1)

        return {
            "text_inputs": {
                "input_ids": text["input_ids"].squeeze(0),
                "attention_mask": text["attention_mask"].squeeze(0),
            },
            "video_frames": video,
            "audio_features": audio,
            "emotion_label": torch.tensor(emotion, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment, dtype=torch.long),
        }

    # =========================
    # COLLATE (FIXED)
    # =========================
    @staticmethod
    def collate_fn(batch):
        # 🔥 FILTER OUT BAD SAMPLES
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return None

        text_inputs = {
            "input_ids": torch.stack([b["text_inputs"]["input_ids"] for b in batch]),
            "attention_mask": torch.stack(
                [b["text_inputs"]["attention_mask"] for b in batch]
            ),
        }

        video = torch.stack([b["video_frames"] for b in batch])
        emotion = torch.stack([b["emotion_label"] for b in batch])
        sentiment = torch.stack([b["sentiment_label"] for b in batch])

        audios = [b["audio_features"] for b in batch]
        max_t = max(a.shape[-1] for a in audios)

        audio = torch.stack([
            torch.nn.functional.pad(a, (0, max_t - a.shape[-1]))
            for a in audios
        ])

        return {
            "text_inputs": text_inputs,
            "video_frames": video,
            "audio_features": audio,
            "emotion_label": emotion,
            "sentiment_label": sentiment,
        }

    # =========================
    # DATALOADERS
    # =========================
    @staticmethod
    def prepare_dataloaders(
        train_csv,
        train_video_dir,
        dev_csv,
        dev_video_dir,
        test_csv,
        test_video_dir,
        batch_size=32,
    ):
        train = DataLoader(
            MeldDataset(train_csv, train_video_dir),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MeldDataset.collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        dev = DataLoader(
            MeldDataset(dev_csv, dev_video_dir),
            batch_size=batch_size,
            collate_fn=MeldDataset.collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        test = DataLoader(
            MeldDataset(test_csv, test_video_dir),
            batch_size=batch_size,
            collate_fn=MeldDataset.collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        return train, dev, test