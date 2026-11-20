"""Run quality screening and classification for one image."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from .quality import assess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    quality = assess_image(image)
    if not quality.accepted:
        print(json.dumps({"status": "rejected_for_quality", "quality": quality.to_dict()}, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((checkpoint.get("input_size", 224),) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        probabilities = model(tf(image).unsqueeze(0).to(device)).softmax(1)[0].cpu()
    ranking = sorted(zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    print(json.dumps({"status": "screening_result", "quality": quality.to_dict(),
                      "predictions": [{"class": name, "score": score} for name, score in ranking]}, indent=2))


if __name__ == "__main__":
    main()
