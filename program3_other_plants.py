"""
Program 3 (other plants): convert the MobileNetV4 checkpoints from Program 2 (apple, corn, grape,
pepper) to .tflite with LiteRT Torch (litert-torch) and download them to your browser.
Made to be PASTED INTO A COLAB CELL and run - no flags needed. Edit the SETTINGS block if you like.

Input : /content/trained_models/<plant>/<plant>_mobilenetv4.pth   (made by Program 2, same session)
        In a NEW session they are missing: the script asks ONCE for an upload, select all the
        <plant>_mobilenetv4.zip files that Program 2 downloaded.
Output: <plant>_mobilenetv4.tflite + <plant>_labels.txt for every plant, in ONE download:
        other_plants_tflite.zip (set ZIP_ALL = False for separate files).
        Unzip it and copy the files into the Pi's AgriLens "models" folder.

The exported model already contains the preprocessing:
  * input : float32, shape [1, 3, 224, 224] (NCHW, RGB), pixel values scaled to 0..1
  * output: float32 [1, num_classes] = softmax probabilities,
            class order = line order in <plant>_labels.txt

HOW TO RUN:
  Paste this whole file into one cell and run it (a GPU is NOT needed for this step).
  If it stops with a message about restarting: Runtime > Restart session, then run the cell again.
  Your files in /content are kept when you restart the session.
"""
import argparse
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

# ============================== SETTINGS (edit here) ==============================
PLANTS = ["apple", "corn", "grape", "pepper"]   # plants to export
DOWNLOAD = True                                 # send the result to your browser
ZIP_ALL = True                                  # True: one zip with everything; False: separate files
# ===================================================================================

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
        torch_before = sys.modules["torch"].__version__.split("+")[0] if "torch" in sys.modules else None
        print(f"Installing missing dependencies: {missing}  (litert-torch is large, ~2 GB, be patient)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
        if torch_before is not None:
            import importlib.metadata as md
            if md.version("torch").split("+")[0] != torch_before:
                sys.exit("The install changed the torch version, so this session must restart.\n"
                         "Runtime > Restart session, then run this cell again (your files are kept).")
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


def ensure_checkpoints(plants, ckpt_dir: Path):
    """Unzip any <plant>_mobilenetv4.zip that is around; if some are missing, ask for ONE upload."""
    def unzip_found():
        for plant in plants:
            ckpt = ckpt_dir / plant / f"{plant}_mobilenetv4.pth"
            if ckpt.exists():
                continue
            for z in (Path("/content") / f"{plant}_mobilenetv4.zip", Path.cwd() / f"{plant}_mobilenetv4.zip",
                      ckpt_dir / f"{plant}_mobilenetv4.zip"):
                if z.exists():
                    print(f"Unzipping {z}")
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(ckpt.parent)
                    break

    unzip_found()
    missing = [p for p in plants if not (ckpt_dir / p / f"{p}_mobilenetv4.pth").exists()]
    if missing and in_colab():
        from google.colab import files
        print(f"\nCheckpoints not found for: {missing}\n"
              f"Select ALL of their zips at once ({', '.join(p + '_mobilenetv4.zip' for p in missing)}):")
        files.upload()  # saved in the current folder (/content)
        unzip_found()


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
    ckpt_dir, out_dir = base / "trained_models", base / "tflite_models"
    plants = [p.lower() for p in PLANTS]
    ensure_checkpoints(plants, ckpt_dir)

    done = []
    for plant in plants:
        ckpt_path = ckpt_dir / plant / f"{plant}_mobilenetv4.pth"
        if not ckpt_path.exists():
            print(f"[{plant}] checkpoint not found -> skipped.")
            continue
        done.append(export_plant(plant, ckpt_path, out_dir, None))

    print(f"\nDone. Exported {len(done)} model(s) to {out_dir.resolve()}")
    if not done or not DOWNLOAD:
        return
    if ZIP_ALL:
        zip_path = out_dir / "other_plants_tflite.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for tflite_path, labels_path in done:
                zf.write(tflite_path, tflite_path.name)
                zf.write(labels_path, labels_path.name)
        print(f"Zipped {len(done)} plant(s) -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        download_to_browser(zip_path)
    else:
        for tflite_path, labels_path in done:
            download_to_browser(tflite_path)
            download_to_browser(labels_path)
    print("Unzip it and copy the .tflite + _labels.txt files into the AgriLens 'models' folder on the Pi.")


if __name__ == "__main__":
    main()
