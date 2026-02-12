import torch
from torch.utils.tensorboard import SummaryWriter
from training.models import MultiModalSentimentModel, MultimodalTrainer


class DummyLoader:
    """Minimal loader to satisfy trainer init"""
    def __len__(self):
        return 1

    @property
    def dataset(self):
        return [0]


def test_logging():
    model = MultiModalSentimentModel()

    dummy_loader = DummyLoader()
    trainer = MultimodalTrainer(model, dummy_loader, dummy_loader)

    # ---- simulate training logging ----
    train_losses = {
        "total": 2.5,
        "emotion": 1.0,
        "sentiment": 1.5
    }

    trainer.log_metrics(train_losses, phase="train")

    # ---- simulate validation logging ----
    val_losses = {
        "total": 1.5,
        "emotion": 0.5,
        "sentiment": 1.0
    }

    val_metrics = {
        "emotion_precision": 0.65,
        "emotion_accuracy": 0.75,
        "sentiment_precision": 0.85,
        "sentiment_accuracy": 0.95
    }

    trainer.log_metrics(val_losses, val_metrics, phase="val")

    print("Logging test passed ✔")


if __name__ == "__main__":
    test_logging()
