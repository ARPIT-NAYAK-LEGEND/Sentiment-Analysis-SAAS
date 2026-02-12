import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torchvision.models.video import R3D_18_Weights

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for p in self.bert.parameters():
            p.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.project(outputs.pooler_output)


# =========================
# VIDEO ENCODER
# =========================
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)

        for p in self.backbone.parameters():
            p.requires_grad = False

        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


# =========================
# AUDIO ENCODER
# =========================
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for p in self.conv.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)
        feats = self.conv(x)
        return self.projection(feats.squeeze(-1))


# =========================
# MULTIMODAL MODEL
# =========================
class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 512 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, text_inputs, video_frames, audio_features):
        t = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )
        v = self.video_encoder(video_frames)
        a = self.audio_encoder(audio_features)

        fused = self.fusion_layer(torch.cat([t, v, a], dim=1))

        return {
            "emotions": self.emotion_classifier(fused),
            "sentiments": self.sentiment_classifier(fused)
        }