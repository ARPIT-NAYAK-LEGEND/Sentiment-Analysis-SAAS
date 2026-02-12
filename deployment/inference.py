import json
import os
import torch
from transformers import AutoTokenizer
from training.models import MultiModalSentimentModel

EMOTION_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):

    model = MultiModalSentimentModel().to(DEVICE)

    model_path = os.path.join(model_dir, "model.pth")
    checkpoint_path = os.path.join(model_dir, "checkpoint.pth")

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=DEVICE)
    elif os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        state = checkpoint["model_state"]
    else:
        raise FileNotFoundError("model.pth or checkpoint.pth not found")

    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    return {
        "model": model,
        "tokenizer": tokenizer
    }

def input_fn(request_body, request_content_type):

    if request_content_type != "application/json":
        raise ValueError("Only application/json supported")

    data = json.loads(request_body)
    return data


def predict_fn(data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    text_inputs = tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    text_inputs = {
        k: v.to(DEVICE) for k, v in text_inputs.items()
    }

    # -------- VIDEO --------
    video_frames = torch.tensor(
        data["video"], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)   # (1, T, C, H, W)

    # -------- AUDIO --------
    audio_features = torch.tensor(
        data["audio"], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)   # (1, 1, 64, T)

    # -------- INFERENCE --------
    with torch.inference_mode():
        outputs = model(
            text_inputs=text_inputs,
            video_frames=video_frames,
            audio_features=audio_features
        )

        emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

    emotion_idx = emotion_probs.argmax().item()
    sentiment_idx = sentiment_probs.argmax().item()

    return {
        "emotion": {
            "label": EMOTION_MAP[emotion_idx],
            "confidence": float(emotion_probs[emotion_idx])
        },
        "sentiment": {
            "label": SENTIMENT_MAP[sentiment_idx],
            "confidence": float(sentiment_probs[sentiment_idx])
        }
    }

def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError("Only application/json supported")

    return json.dumps(prediction)
