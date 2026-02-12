import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torchvision.models.video import R3D_18_Weights
from sklearn.metrics import precision_score, accuracy_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

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


# =========================
# CLASS WEIGHTS (FIXED)
# =========================
def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)

    print("Using pre-computed class distributions (fast mode)...")
    # Values from previous run (dataset/train/train_splits)
    # anger: 1109, disgust: 271, fear: 268, happy: 1743, neutral: 4710, sad: 683, surprise: 1205
    # negative: 2945, neutral: 4710, positive: 2334
    
    emotion_counts = torch.tensor([1109., 271., 268., 1743., 4710., 683., 1205.])
    sentiment_counts = torch.tensor([2945., 4710., 2334.])
    skipped = 0
    total = 9989

    valid = total - skipped
    print(f"Skipped {skipped}/{total} samples due to loading errors.")

    print("\nClass distribution:")
    emotion_map = {
        0: "anger", 1: "disgust", 2: "fear",
        3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
    }
    for i, c in enumerate(emotion_counts):
        print(f"  {emotion_map[i]}: {int(c)} ({(c/valid)*100:.2f}%)")

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    for i, c in enumerate(sentiment_counts):
        print(f"  {sentiment_map[i]}: {int(c)} ({(c/valid)*100:.2f}%)")

    emotion_weights = 1.0 / torch.clamp(emotion_counts, min=1)
    sentiment_weights = 1.0 / torch.clamp(sentiment_counts, min=1)

    emotion_weights /= emotion_weights.sum()
    sentiment_weights /= sentiment_weights.sum()

    return emotion_weights, sentiment_weights


# =========================
# TRAINER
# =========================
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        self.writer = SummaryWriter(f"{base_dir}/run_{timestamp}")

        self.global_step = 0

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-4,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=2
        )

        print("Calculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)
        device = next(model.parameters()).device

        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=emotion_weights.to(device),
            label_smoothing=0.05
        )
        self.sentiment_criterion = nn.CrossEntropyLoss(
            weight=sentiment_weights.to(device),
            label_smoothing=0.05
        )

    # =========================
    # TRAIN
    # =========================
    def train_epoch(self):
        self.model.train()
        running = {"total": 0, "emotion": 0, "sentiment": 0}
        valid_batches = 0
        device = next(self.model.parameters()).device

        # Iterate over batches
        for i, batch in enumerate(self.train_loader):
            if batch is None:
                continue

            if i % 5 == 0:
                print(f"[Epoch running] batch {i}")


            valid_batches += 1

            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }

            outputs = self.model(
                text_inputs,
                batch["video_frames"].to(device),
                batch["audio_features"].to(device)
            )

            e_loss = self.emotion_criterion(outputs["emotions"], batch["emotion_label"].to(device))
            s_loss = self.sentiment_criterion(outputs["sentiments"], batch["sentiment_label"].to(device))
            loss = e_loss + s_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            running["total"] += loss.item()
            running["emotion"] += e_loss.item()
            running["sentiment"] += s_loss.item()

            self.global_step += 1

        return {k: v / max(1, valid_batches) for k, v in running.items()}

    # =========================
    # EVALUATE
    # =========================
    def evaluate(self, loader, phase="val"):
        self.model.eval()
        losses = {"total": 0, "emotion": 0, "sentiment": 0}
        valid_batches = 0

        e_preds, s_preds, e_lbls, s_lbls = [], [], [], []
        device = next(self.model.parameters()).device

        with torch.inference_mode():
            for batch in loader:
                if batch is None:
                    continue

                valid_batches += 1

                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }

                outputs = self.model(
                    text_inputs,
                    batch["video_frames"].to(device),
                    batch["audio_features"].to(device)
                )

                e_loss = self.emotion_criterion(outputs["emotions"], batch["emotion_label"].to(device))
                s_loss = self.sentiment_criterion(outputs["sentiments"], batch["sentiment_label"].to(device))
                loss = e_loss + s_loss

                losses["total"] += loss.item()
                losses["emotion"] += e_loss.item()
                losses["sentiment"] += s_loss.item()

                e_preds.extend(outputs["emotions"].argmax(1).cpu().numpy())
                s_preds.extend(outputs["sentiments"].argmax(1).cpu().numpy())
                e_lbls.extend(batch["emotion_label"].cpu().numpy())
                s_lbls.extend(batch["sentiment_label"].cpu().numpy())

        avg_losses = {k: v / max(1, valid_batches) for k, v in losses.items()}

        metrics = {
            "emotion_accuracy": accuracy_score(e_lbls, e_preds),
            "emotion_precision": precision_score(e_lbls, e_preds, average="weighted"),
            "sentiment_accuracy": accuracy_score(s_lbls, s_preds),
            "sentiment_precision": precision_score(s_lbls, s_preds, average="weighted")
        }

        if phase == "val":
            self.scheduler.step(avg_losses["total"])

        return avg_losses, metrics