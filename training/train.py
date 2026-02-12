import torch
from pathlib import Path
import argparse
import os

from training.meld_dataset import MeldDataset
from training.models import MultiModalSentimentModel, MultimodalTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Sentiment Model")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--limit_check", action="store_true")

    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    args = parser.parse_args()

    # ===================== DEVICE =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ===================== MODEL DIR =====================
    model_dir = os.environ.get("SM_MODEL_DIR", "./model")
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, "checkpoint.pth")

    # ===================== DATA PATHS =====================
    repo_root = Path(__file__).parent.parent

    train_dir = Path(args.train_dir) if args.train_dir else repo_root / "dataset/train"
    train_csv = train_dir / "train_sent_emo.csv"
    train_video_dir = train_dir / "train_splits"

    val_dir = Path(args.val_dir) if args.val_dir else repo_root / "dataset/dev"
    dev_csv = val_dir / "dev_sent_emo.csv"
    dev_video_dir = val_dir / "dev_splits_complete"

    test_dir = Path(args.test_dir) if args.test_dir else repo_root / "dataset/test"
    test_csv = test_dir / "test_sent_emo.csv"
    test_video_dir = test_dir / "output_repeated_splits_test"

    for path in [train_csv, dev_csv, test_csv]:
        if not path.exists():
            raise FileNotFoundError(f"❌ Missing required file: {path}")

    print("✅ All dataset files found")

    # ===================== LOADERS =====================
    print("Loading datasets...")
    train_loader, dev_loader, test_loader = MeldDataset.prepare_dataloaders(
        str(train_csv), str(train_video_dir),
        str(dev_csv), str(dev_video_dir),
        str(test_csv), str(test_video_dir),
        batch_size=args.batch_size
    )

    if args.limit_check:
        print("⚠ Running limit_check mode")
        indices = list(range(args.batch_size * 2))

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, indices),
            batch_size=args.batch_size,
            collate_fn=MeldDataset.collate_fn
        )

        dev_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dev_loader.dataset, indices),
            batch_size=args.batch_size,
            collate_fn=MeldDataset.collate_fn
        )

    # ===================== MODEL =====================
    print("Initializing model...")
    model = MultiModalSentimentModel().to(device)

    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader
    )

    # ===================== RESUME LOGIC =====================
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        print("🔁 Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"✅ Resumed from epoch {start_epoch}")

    # ===================== TRAIN LOOP =====================
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = trainer.train_epoch()
        print(f"Train Loss: {train_metrics['total']:.4f}")

        val_losses, val_metrics = trainer.evaluate(dev_loader, phase="val")
        print(
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Acc (Emo): {val_metrics['emotion_accuracy']:.4f}"
        )
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_path = os.path.join(model_dir, "model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved best model to {best_path}")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"📦 Checkpoint saved (epoch {epoch + 1})")

        if args.limit_check:
            print("Limit check complete.")
            break

    print("🎉 Training complete")


if __name__ == "__main__":
    main()
