"""
MANGO: download dataset -> train MobileNetV4 -> export .tflite  (single script, Colab-ready)

Dataset : https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset (MangoLeafBD)
          Checked against the original Mendeley record and a Hugging Face copy: 4000 JPG images
          (about 240x320), 8 classes x 500 images: Anthracnose, Bacterial Canker, Cutting Weevil,
          Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould.
          CAVEAT: published papers describe only ~1800 distinct leaves; the rest are constructed
          (augmented) copies. A random split can put copies of the same leaf in train AND test, so the
          accuracy printed here is probably optimistic. Test on real leaves before trusting it.
Output  : mango_mobilenetv4.tflite + mango_labels.txt  -> downloaded to your browser.
          Same format as the tomato/potato models, so they drop straight into the AgriLens Pi app
          (put both files in its models/ folder). Input: float32 [1,3,224,224] RGB 0..1,
          output: softmax probabilities in the order of mango_labels.txt.

COLAB USAGE:
  1. Runtime > Change runtime type > GPU (T4)
  2. Upload this file, then run:   %run mango_train_export.py
     (best in a FRESH runtime: installing litert-torch can upgrade torch. If you see a
      version/import error right after the install, Runtime > Restart session and run again.)
  Kaggle login: the script uses your Kaggle USERNAME + KEY. It reads them from the environment
  (KAGGLE_USERNAME / KAGGLE_KEY) or from Colab Secrets with those names; if neither exists and the
  download needs a login, it asks you to type them (the key is hidden while typing).
  Last resort: it asks for your kaggle.json file.

  Optional flags: --epochs 15  --target_acc 0.96 (stop once val accuracy reaches 96%)
                  --source huggingface (skip Kaggle, use the login-free copy)
                  --data_dir /path/to/extracted/dataset  --no_download
"""
import argparse
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_SLUG = "aryashah2k/mango-leaf-disease-dataset"
HF_SLUG = "Project-AgML/MangoLeafBD_disease_classification"  # same dataset, no login needed
EXPECTED_PER_CLASS = 500
PLANT = "mango"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_PACKAGES = {  # import name -> pip name
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "pillow",
    "timm": "timm",
    "kagglehub": "kagglehub",
    "litert_torch": "litert-torch",
    "ai_edge_litert": "ai-edge-litert",
}


def install_dependencies():
    """System command: pip-install any missing dependency."""
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing dependencies: {missing}  (litert-torch is large, be patient)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    else:
        print("All dependencies already installed.")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


install_dependencies()

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import litert_torch


def download_to_browser(path: Path):
    if in_colab():
        from google.colab import files
        files.download(str(path))
    else:
        print(f"(not in Colab) file saved at: {path}")


# ----------------------------------------------------------------------------
# 1) Download the dataset
# ----------------------------------------------------------------------------
def set_kaggle_credentials(ask: bool) -> bool:
    """Make KAGGLE_USERNAME / KAGGLE_KEY available (works for kagglehub and the kaggle CLI)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if in_colab():  # Colab 'Secrets' (key icon in the left bar) named KAGGLE_USERNAME / KAGGLE_KEY
        try:
            from google.colab import userdata
            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
            return True
        except Exception:  # noqa: BLE001
            pass
    if not ask:
        return False
    import getpass
    user = input("Kaggle username: ").strip()
    keyval = getpass.getpass("Kaggle key (hidden): ").strip()
    if not user or not keyval:
        return False
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, keyval
    return True


def download_from_huggingface() -> Path:
    """Login-free copy of the same dataset (Project-AgML/MangoLeafBD_disease_classification)."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
    from datasets import load_dataset
    print("Downloading the Hugging Face copy ...")
    ds = load_dataset(HF_SLUG)
    out = Path("/content/mango_raw_hf") if in_colab() else Path("mango_raw_hf")
    n = 0
    for sname, split in ds.items():
        names = split.features["label"].names
        for i, row in enumerate(split):
            d = out / names[row["label"]]
            d.mkdir(parents=True, exist_ok=True)
            row["image"].convert("RGB").save(d / f"{sname}_{i}.jpg", quality=95)
            n += 1
    print(f"Saved {n} images to {out}")
    return out


def download_dataset(args) -> Path:
    if args.data_dir:
        return args.data_dir
    if args.source == "huggingface":
        return download_from_huggingface()
    import kagglehub
    set_kaggle_credentials(ask=False)  # use env vars / Colab secrets if they exist
    for attempt in (1, 2):
        try:
            print("Downloading dataset with kagglehub ...")
            return Path(kagglehub.dataset_download(KAGGLE_SLUG))
        except Exception as e:  # noqa: BLE001
            print(f"Download failed: {e}")
            if attempt == 1:
                print("\nEnter your Kaggle username and key:")
                if not set_kaggle_credentials(ask=True):
                    break
    try:  # Kaggle did not work: the same dataset is also on Hugging Face without any login
        print("Trying the Hugging Face copy of the dataset instead ...")
        return download_from_huggingface()
    except Exception as e:  # noqa: BLE001
        print(f"Hugging Face download failed too: {e}")
    # last resort: kaggle.json upload
    if not in_colab():
        sys.exit("Set KAGGLE_USERNAME / KAGGLE_KEY, or pass --data_dir with the extracted dataset.")
    from google.colab import files
    print("Upload your kaggle.json instead (kaggle.com > Settings > API > Create New Token):")
    up = files.upload()
    if "kaggle.json" not in up:
        sys.exit("Could not download the dataset.")
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    (kdir / "kaggle.json").write_bytes(up["kaggle.json"])
    (kdir / "kaggle.json").chmod(0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    out = Path("/content/mango_raw")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(out), "--unzip"], check=True)
    return out


def find_class_root(root: Path) -> Path:
    """The folder whose sub-folders are the class folders (the one with the most image folders)."""
    best, best_n = None, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        n = sum(1 for c in d.iterdir()
                if c.is_dir() and any(f.suffix.lower() in IMG_EXT for f in c.iterdir() if f.is_file()))
        if n > best_n:
            best, best_n = d, n
    if best is None or best_n < 2:
        sys.exit(f"Could not find class folders with images inside {root}")
    return best


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def make_splits(class_root: Path, val_frac, test_frac, seed):
    rng = random.Random(seed)
    classes = sorted([d for d in class_root.iterdir() if d.is_dir() and
                      any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())],
                     key=lambda d: key(d.name))
    names = [key(d.name) for d in classes]
    splits = {"train": [], "val": [], "test": []}
    for idx, d in enumerate(classes):
        files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        n_test, n_val = max(1, int(len(files) * test_frac)), max(1, int(len(files) * val_frac))
        splits["test"] += [(f, idx) for f in files[:n_test]]
        splits["val"] += [(f, idx) for f in files[n_test:n_test + n_val]]
        splits["train"] += [(f, idx) for f in files[n_test + n_val:]]
        flag = "" if len(files) == EXPECTED_PER_CLASS else f"   <- expected {EXPECTED_PER_CLASS}"
        print(f"   {names[idx]:18s} {len(files):4d} images{flag}")
    return names, splits


class ImgDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


# ----------------------------------------------------------------------------
# 2) Train
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    model.eval()
    correct, total = np.zeros(n_cls), np.zeros(n_cls)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            total[t] += 1
            correct[t] += int(t == p)
    return correct.sum() / max(total.sum(), 1), correct, total


def train(args, names, splits, device):
    model = timm.create_model(args.model, pretrained=True)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]
    size = args.img_size

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05), transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(ImgDataset(splits["train"], train_tf), args.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(ImgDataset(splits["val"], eval_tf), args.batch_size, **kw)
    test_dl = DataLoader(ImgDataset(splits["test"], eval_tf), args.batch_size, **kw)
    print(f"Images: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    model.reset_classifier(num_classes=len(names))
    model.to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr * 0.2}, {"params": head, "lr": args.lr}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_state, bad = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
            n += len(x)
        sched.step()
        val_acc, _, _ = evaluate(model, val_dl, device, len(names))
        print(f"epoch {epoch:2d}/{args.epochs}  loss {run / n:.4f}  val_acc {val_acc:.4f}  ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if val_acc >= args.target_acc and epoch >= args.min_epochs:
                print(f"Reached target val accuracy ({val_acc:.4f} >= {args.target_acc:.2f}) -> stopping training.")
                break
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    model.load_state_dict(best_state)
    test_acc, correct, total = evaluate(model, test_dl, device, len(names))
    print(f"\nBest val acc: {best_acc:.4f} | TEST acc: {test_acc:.4f}")
    for i, c in enumerate(names):
        print(f"   {c:18s} {int(correct[i]):3d}/{int(total[i]):3d}")
    return model.cpu().eval(), mean, std


# ----------------------------------------------------------------------------
# 3) Export to TFLite with litert-torch
# ----------------------------------------------------------------------------
class DeployModel(nn.Module):
    """Input 0..1 RGB NCHW -> normalisation -> MobileNetV4 -> softmax."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.softmax(self.net((x - self.mean) / self.std), dim=1)


def first_output(out):
    if isinstance(out, (list, tuple)):
        out = out[0]
    return np.asarray(out)


def export_tflite(net, mean, std, names, size, out_dir: Path):
    model = DeployModel(net, mean, std).eval()
    sample = (torch.rand(1, 3, size, size),)
    with torch.no_grad():
        torch_out = model(*sample).numpy()
    print("Converting with litert-torch ...")
    edge_model = litert_torch.convert(model, sample)
    diff = float(np.abs(torch_out - first_output(edge_model(*sample))).max())
    print(f"Max |PyTorch - LiteRT| difference: {diff:.6f} ({'OK' if diff < 1e-2 else 'WARNING: larger than expected'})")

    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / f"{PLANT}_mobilenetv4.tflite"
    labels_path = out_dir / f"{PLANT}_labels.txt"
    edge_model.export(str(tflite_path))
    labels_path.write_text("\n".join(names) + "\n")
    print(f"Saved -> {tflite_path} ({tflite_path.stat().st_size / 1e6:.2f} MB)")
    try:  # reload check (non-fatal)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=str(tflite_path))
        it.allocate_tensors()
        i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i0["index"], sample[0].numpy().astype(np.float32))
        it.invoke()
        print(f"Reloaded .tflite check: max diff {float(np.abs(torch_out - it.get_tensor(o0['index'])).max()):.6f}"
              f" | input {i0['shape'].tolist()} | output {o0['shape'].tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped reload check: {e})")
    return tflite_path, labels_path


def main():
    base = Path("/content") if in_colab() else Path(".")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=None, help="already-extracted dataset folder")
    ap.add_argument("--source", choices=["kaggle", "huggingface"], default="kaggle")
    ap.add_argument("--out_dir", type=Path, default=base / "mango_output")
    ap.add_argument("--model", default="mobilenetv4_conv_small.e2400_r224_in1k")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target_acc", type=float, default=0.96, help="stop once val accuracy reaches this")
    ap.add_argument("--min_epochs", type=int, default=1, help="don't stop on target_acc before this epoch")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_download", action="store_true", help="don't send the files to the browser")
    args, _ = ap.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (slow! Runtime > Change runtime type > GPU)"))

    root = download_dataset(args)
    class_root = find_class_root(root)
    print(f"\nClass folders found in: {class_root}")
    names, splits = make_splits(class_root, args.val_frac, args.test_frac, args.seed)
    print(f"Classes ({len(names)}): {names}")
    if len(names) != 8:
        print("NOTE: expected 8 mango classes; check the dataset folder if this looks wrong.")

    net, mean, std = train(args, names, splits, device)
    tflite_path, labels_path = export_tflite(net, mean, std, names, args.img_size, args.out_dir)

    if not args.no_download:
        download_to_browser(tflite_path)
        download_to_browser(labels_path)
    print("\nDone. Copy mango_mobilenetv4.tflite + mango_labels.txt into the AgriLens 'models' folder on the Pi.")


if __name__ == "__main__":
    main()
