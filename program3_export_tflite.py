"""
Program 3: Convert the per-plant MobileNetV4 checkpoints from Program 2 to .tflite
using LiteRT Torch (litert-torch), then download them to your browser. Colab-ready.

Input : <ckpt_dir>/<plant>/<plant>_mobilenetv4.pth   (made by Program 2)
        If missing, it looks for <plant>_mobilenetv4.zip (what Program 2 downloaded)
        and, in Colab, asks you to upload it.
Output: <plant>_mobilenetv4.tflite  +  <plant>_labels.txt  -> downloaded to your browser.

The exported model already contains the preprocessing, so in your app you only need to:
  * input : float32, shape [1, 3, H, W] (NCHW, RGB), pixel values scaled to 0..1
            (H = W = 224 by default)
  * output: float32 [1, num_classes] = softmax probabilities,
            class order = line order in <plant>_labels.txt

COLAB USAGE:
  Upload this file, then run:   %run program3_export_tflite.py
  Optional flags: --plants tomato potato   --no_download
  Note: installing litert-torch can upgrade torch. If you get an import/version error
  right after the install, use Runtime > Restart session and run this again.
"""
import argparse
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

REQUIRED_PACKAGES = {  # import name -> pip name
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "timm": "timm",
    "litert_torch": "litert-torch",
    "ai_edge_litert": "ai-edge-litert",
}


def install_dependencies():
    """System command: pip-install any missing dependency."""
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing dependencies: {missing}  (litert-torch is large, ~2 GB, be patient)")
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
import litert_torch


def download_to_browser(path: Path):
    if in_colab():
        from google.colab import files
        files.download(str(path))
    else:
        print(f"(not in Colab) file saved at: {path}")


def locate_checkpoint(plant: str, ckpt_dir: Path):
    ckpt = ckpt_dir / plant / f"{plant}_mobilenetv4.pth"
    if ckpt.exists():
        return ckpt
    zips = [Path("/content") / f"{plant}_mobilenetv4.zip", Path.cwd() / f"{plant}_mobilenetv4.zip",
            ckpt_dir / f"{plant}_mobilenetv4.zip"]

    def try_unzip():
        for z in zips:
            if z.exists():
                print(f"Unzipping {z}")
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(ckpt.parent)
                return ckpt.exists()
        return False

    if try_unzip():
        return ckpt
    if in_colab():
        from google.colab import files
        print(f"\nCheckpoint for '{plant}' not found. Upload '{plant}_mobilenetv4.zip' (from Program 2):")
        files.upload()
        if try_unzip():
            return ckpt
    return None


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


def export_plant(plant, ckpt_path, out_dir, args):
    print(f"\n{'=' * 60}\nExporting TFLite for: {plant}\n{'=' * 60}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes, size = ckpt["classes"], ckpt["img_size"]
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Checkpoint val_acc={ckpt.get('val_acc', float('nan')):.4f}  "
          f"test_acc={ckpt.get('test_acc', float('nan')):.4f}")

    net = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(classes))
    net.load_state_dict(ckpt["state_dict"])
    model = DeployModel(net, ckpt["mean"], ckpt["std"]).eval()

    sample = (torch.rand(1, 3, size, size),)
    with torch.no_grad():
        torch_out = model(*sample).numpy()

    print("Converting with litert-torch ...")
    edge_model = litert_torch.convert(model, sample)

    # Check 1: converted model vs PyTorch
    edge_out = first_output(edge_model(*sample))
    diff = float(np.abs(torch_out - edge_out).max())
    print(f"Max |PyTorch - LiteRT| difference: {diff:.6f}  "
          f"({'OK' if diff < 1e-2 else 'WARNING: larger than expected'})")

    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / f"{plant}_mobilenetv4.tflite"
    edge_model.export(str(tflite_path))
    labels_path = out_dir / f"{plant}_labels.txt"
    labels_path.write_text("\n".join(classes) + "\n")
    print(f"Saved -> {tflite_path} ({tflite_path.stat().st_size / 1e6:.2f} MB)")

    # Check 2: reload the exported .tflite file with the LiteRT interpreter (non-fatal)
    try:
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=str(tflite_path))
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(inp["index"], sample[0].numpy().astype(np.float32))
        it.invoke()
        file_out = it.get_tensor(outp["index"])
        print(f"Reloaded .tflite check: max diff {float(np.abs(torch_out - file_out).max()):.6f} "
              f"| input {inp['shape'].tolist()} | output {outp['shape'].tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped reload check: {e})")

    return tflite_path, labels_path


def main():
    base = Path("/content") if in_colab() else Path(".")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=Path, default=base / "trained_models")
    ap.add_argument("--out_dir", type=Path, default=base / "tflite_models")
    ap.add_argument("--plants", nargs="+", default=None,
                    help="default: every plant checkpoint found in ckpt_dir, else tomato potato")
    ap.add_argument("--no_download", action="store_true")
    args, _ = ap.parse_known_args()

    plants = args.plants
    if plants is None:
        found = sorted(p.name for p in args.ckpt_dir.glob("*")
                       if (p / f"{p.name}_mobilenetv4.pth").exists()) if args.ckpt_dir.exists() else []
        plants = found or ["tomato", "potato"]

    done = []
    for plant in plants:
        ckpt_path = locate_checkpoint(plant.lower(), args.ckpt_dir)
        if ckpt_path is None:
            print(f"[{plant}] checkpoint not found -> skipped.")
            continue
        tflite_path, labels_path = export_plant(plant.lower(), ckpt_path, args.out_dir, args)
        done.append((tflite_path, labels_path))
        if not args.no_download:
            download_to_browser(tflite_path)
            download_to_browser(labels_path)

    print(f"\nDone. Exported {len(done)} model(s) to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
