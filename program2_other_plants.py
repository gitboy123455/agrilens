"""
Program 2 (other plants): train ONE MobileNetV4 model per plant (apple, corn, grape, pepper).
PyTorch + timm. Made to be PASTED INTO A COLAB CELL and run - no command-line flags needed.
Change the SETTINGS block just below the imports if you want something different.

Input : the per-plant datasets built by program1_other_plants.py
        (/content/plant_datasets/<plant>/{train,val,test}/<class>/*.jpg).
        If you are in a NEW Colab session they are missing: the script asks ONCE for an upload,
        select all the <plant>_dataset.zip files that Program 1 downloaded (apple, corn, grape, pepper).
Output: <plant>_mobilenetv4.zip  (checkpoint .pth + class list + log) downloaded to your browser.
        Program 3 turns these into .tflite files.

HOW TO RUN:
  1. Runtime > Change runtime type > GPU (T4 is fine)
  2. Paste this whole file into one cell and run it.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ============================== SETTINGS (edit here) ==============================
PLANTS = ["apple", "corn", "grape", "pepper"]   # one model is trained per plant in this list
EPOCHS = 15                                     # maximum epochs per plant
TARGET_ACC = 0.96                               # stop once validation accuracy reaches 96%
MIN_EPOCHS = 3                                  # ...but never stop before this epoch (epoch 1 is too early)
MODEL = "mobilenetv4_conv_small.e2400_r224_in1k"
# ===================================================================================


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


REQUIRED_PACKAGES = {  # import name -> pip name
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "timm": "timm",
}


def install_dependencies():
    """System command: pip-install any missing dependency (Colab already ships most of them)."""
    import importlib.util
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing dependencies: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    else:
        print("All dependencies already installed.")


install_dependencies()

import timm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def download_to_browser(path: Path):
    if in_colab():
        from google.colab import files
        files.download(str(path))
    else:
        print(f"(not in Colab) file saved at: {path}")


def locate_dataset(plant: str, data_dir: Path):
    """Find <data_dir>/<plant>, unzipping <plant>_dataset.zip or asking for an upload if needed."""
    d = data_dir / plant
    if (d / "train").exists():
        return d
    zips = [Path("/content") / f"{plant}_dataset.zip", Path.cwd() / f"{plant}_dataset.zip",
            data_dir / f"{plant}_dataset.zip"]

    def try_unzip():
        for z in zips:
            if z.exists():
                print(f"Unzipping {z} -> {d}")
                d.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
                return True
        return False

    if try_unzip():
        return d
    if in_colab():
        from google.colab import files
        print(f"\nDataset for '{plant}' not found. Upload '{plant}_dataset.zip' (from Program 1):")
        files.upload()
        if try_unzip():
            return d
    return None


def make_loaders(ds_dir: Path, img_size: int, mean, std, batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(ds_dir / "train", train_tf)
    val_ds = datasets.ImageFolder(ds_dir / "val", eval_tf)
    test_ds = datasets.ImageFolder(ds_dir / "test", eval_tf)
    assert train_ds.classes == val_ds.classes == test_ds.classes, "class folders differ between splits"

    # moderate class balancing (PlantDoc-only classes are often much smaller)
    counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
    w_cls = 1.0 / np.sqrt(np.maximum(counts, 1))
    sampler = WeightedRandomSampler(w_cls[train_ds.targets], num_samples=len(train_ds), replacement=True)

    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    return (train_ds,
            DataLoader(train_ds, batch_size, sampler=sampler, drop_last=True, **kw),
            DataLoader(val_ds, batch_size, shuffle=False, **kw),
            DataLoader(test_ds, batch_size, shuffle=False, **kw))


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y.numpy(), pred.numpy()):
            total[t] += 1
            correct[t] += int(t == p)
    return correct.sum() / max(total.sum(), 1), correct, total


def train_plant(plant, ds_dir, args, device, out_dir):
    print(f"\n{'=' * 60}\nTraining model for: {plant}\n{'=' * 60}")
    model = timm.create_model(args.model, pretrained=True)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]

    train_ds, train_dl, val_dl, test_dl = make_loaders(
        ds_dir, args.img_size, mean, std, args.batch_size, args.workers)
    classes = train_ds.classes
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Images: train={len(train_ds)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    model.reset_classifier(num_classes=len(classes))
    model.to(device)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr * 0.2},
                             {"params": head_params, "lr": args.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_state, bad_epochs, log = 0.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(x)
            n += len(x)
        sched.step()
        val_acc, _, _ = evaluate(model, val_dl, device, len(classes))
        log.append({"epoch": epoch, "train_loss": run_loss / n, "val_acc": float(val_acc)})
        print(f"epoch {epoch:2d}/{args.epochs}  loss {run_loss / n:.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if val_acc >= args.target_acc and epoch >= args.min_epochs:
                print(f"Reached target val accuracy ({val_acc:.4f} >= {args.target_acc:.2f}) "
                      f"-> stopping training.")
                break
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    model.load_state_dict(best_state)
    test_acc, correct, total = evaluate(model, test_dl, device, len(classes))
    print(f"\nBest val acc: {best_acc:.4f} | TEST acc (PlantDoc real-world images): {test_acc:.4f}")
    for i, c in enumerate(classes):
        print(f"   {c:26s} {int(correct[i]):3d}/{int(total[i]):3d}")

    # ---- save + zip ----
    pdir = out_dir / plant
    pdir.mkdir(parents=True, exist_ok=True)
    ckpt = pdir / f"{plant}_mobilenetv4.pth"
    torch.save({"state_dict": best_state, "model_name": args.model, "classes": classes,
                "img_size": args.img_size, "mean": list(mean), "std": list(std),
                "plant": plant, "val_acc": float(best_acc), "test_acc": float(test_acc)}, ckpt)
    (pdir / f"{plant}_classes.json").write_text(json.dumps(classes, indent=2))
    (pdir / f"{plant}_train_log.json").write_text(json.dumps(
        {"log": log, "best_val_acc": float(best_acc), "test_acc": float(test_acc)}, indent=2))
    zip_path = Path(shutil.make_archive(str(out_dir / f"{plant}_mobilenetv4"), "zip", pdir))
    print(f"Saved -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def ensure_datasets(plants, data_dir: Path):
    """Unzip any <plant>_dataset.zip that is around; if some are still missing, ask for ONE upload."""
    def unzip_found():
        for plant in plants:
            d = data_dir / plant
            if (d / "train").exists():
                continue
            for z in (Path("/content") / f"{plant}_dataset.zip", Path.cwd() / f"{plant}_dataset.zip",
                      data_dir / f"{plant}_dataset.zip"):
                if z.exists():
                    print(f"Unzipping {z} -> {d}")
                    d.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(d)
                    break

    unzip_found()
    missing = [p for p in plants if not (data_dir / p / "train").exists()]
    if missing and in_colab():
        from google.colab import files
        print(f"\nDatasets not found for: {missing}\n"
              f"Select ALL of their zips at once ({', '.join(p + '_dataset.zip' for p in missing)}):")
        files.upload()  # saved in the current folder (/content)
        unzip_found()


def main():
    base = Path("/content") if in_colab() else Path(".")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=base / "plant_datasets")
    ap.add_argument("--out_dir", type=Path, default=base / "trained_models")
    ap.add_argument("--plants", nargs="+", default=PLANTS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_epochs", type=int, default=MIN_EPOCHS)
    ap.add_argument("--target_acc", type=float, default=TARGET_ACC,
                    help="stop training once validation accuracy reaches this (0.96 = 96%%)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_download", action="store_true")
    args, _ = ap.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda"
                                 else "  (slow! In Colab: Runtime > Change runtime type > GPU)"))

    plants = [p.lower() for p in args.plants]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_datasets(plants, args.data_dir)
    for plant in plants:
        ds_dir = args.data_dir / plant if (args.data_dir / plant / "train").exists() else None
        if ds_dir is None:
            print(f"[{plant}] dataset not found -> skipped.")
            continue
        zip_path = train_plant(plant, ds_dir, args, device, args.out_dir)
        if not args.no_download:
            download_to_browser(zip_path)
    print("\nAll done. Next: Program 3 converts these checkpoints to .tflite.")


if __name__ == "__main__":
    main()
