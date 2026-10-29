"""Train and use a LeNet-style smile classifier.

Portfolio revision, October 2026. This repairs the archived architecture and
provides a command-line workflow; it does not reproduce a verified 2023 score.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

from manifest import read_manifest, validate_splits


def preprocessing():
    return transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


class FaceDataset(Dataset):
    def __init__(self, image_dir, rows):
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform = preprocessing()
        missing = [name for name, _ in rows if not (self.image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, label = self.rows[index]
        with Image.open(self.image_dir / name) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


class SmileCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 29 * 29, 120), nn.ReLU(),
            nn.Linear(120, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            size = labels.numel()
            total_loss += loss.item() * size
            correct += (logits.argmax(1) == labels).sum().item()
            count += size
    return {"loss": total_loss / count, "accuracy": correct / count}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(args.device)
    train_rows = read_manifest(args.train_csv)
    valid_rows = read_manifest(args.valid_csv)
    validate_splits(train_rows, valid_rows)
    train_loader = DataLoader(
        FaceDataset(args.image_dir, train_rows), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        FaceDataset(args.image_dir, valid_rows), batch_size=args.batch_size,
    )
    output = Path(args.output)
    # A separate directory per experiment prevents overwriting earlier evidence.
    output.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if key != "handler"}
    config.update({
        "resolved_device": str(device), "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "train_examples": len(train_rows), "valid_examples": len(valid_rows),
    })
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = SmileCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    history, best_loss = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        valid_metrics = run_epoch(model, valid_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": valid_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            # Store tensors and plain metadata rather than a pickled model object.
            torch.save({
                "architecture": "smile-cnn-v1", "epoch": epoch,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "validation": valid_metrics,
            }, output / "best.pt")
    print(f"Saved history and best checkpoint to {output}")


def predict(args):
    device = select_device(args.device)
    image_dir = Path(args.image_dir)
    image_ids = sorted(
        path.name for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_ids:
        raise ValueError(f"No JPEG or PNG images found in {image_dir}")
    loader = DataLoader(
        FaceDataset(image_dir, [(name, 0) for name in image_ids]),
        batch_size=args.batch_size,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "smile-cnn-v1":
        raise ValueError("Expected a checkpoint produced by this pipeline")
    model = SmileCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            probabilities.extend(model(images.to(device)).softmax(1)[:, 1].cpu().tolist())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "smile_probability"])
        writer.writerows(
            (name, int(probability > 0.5), probability)
            for name, probability in zip(image_ids, probabilities)
        )
    print(f"Saved {len(image_ids)} predictions to {output}")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train on image manifests")
    training.add_argument("--image-dir", required=True)
    training.add_argument("--train-csv", required=True)
    training.add_argument("--valid-csv", required=True)
    training.add_argument("--output", default="outputs/baseline")
    training.add_argument("--epochs", type=positive_int, default=20)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--seed", type=int, default=190)
    training.set_defaults(handler=train)
    inference = commands.add_parser("predict", help="Predict on a folder of images")
    inference.add_argument("--image-dir", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", default="outputs/predictions.csv")
    inference.set_defaults(handler=predict)
    for command in [training, inference]:
        command.add_argument("--batch-size", type=positive_int, default=64)
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
