"""
Program 1 (other plants): build per-plant datasets for APPLE, CORN, GRAPE and BELL PEPPER
by merging PlantDoc + PlantVillage (both are GitHub datasets).

Why these four: PlantVillage covers 14 crops and PlantDoc 13. After applying your rules, only
these four keep at least 2 classes:
  * Apple  -> apple_scab, cedar_apple_rust, healthy       (PlantVillage "black rot" thrown out)
  * Corn   -> gray_leaf_spot, common_rust, northern_leaf_blight  (PlantVillage "healthy" thrown out)
  * Grape  -> black_rot, healthy                          (PlantVillage "esca", "leaf blight" thrown out)
  * Pepper -> bacterial_spot, healthy
The others (blueberry, cherry, peach, raspberry, soybean, squash, strawberry) end up with only ONE
class (PlantDoc has just a healthy leaf), so no classifier can be trained -> skipped automatically.
Orange exists only in PlantVillage -> skipped (not in PlantDoc).

Rules (per plant):
  * No class in common between the two datasets -> use PlantDoc only.
  * Some classes in common -> merge PlantDoc + PlantVillage images for those classes.
        - PlantVillage-only classes  -> thrown out.
        - PlantDoc-only classes      -> kept.
  * Plant not found in PlantDoc, or fewer than 2 classes left -> skipped.

Split policy (so test numbers reflect real-world photos):
  * PlantDoc test split  -> test   (PlantDoc only, never PlantVillage)
  * PlantDoc train + PlantVillage -> train / val

COLAB USAGE (upload the file and run it):
  %run program1_other_plants.py
  -> clones PlantDoc + PlantVillage from GitHub automatically, builds the
     per-plant datasets in /content/plant_datasets, zips each plant and
     downloads the zips to your browser.

Optional flags (Colab or local):
  --plantvillage PATH --plantdoc PATH   use your own copies instead of cloning
  --plants apple corn grape pepper
  --no_download                          skip the browser download

NEXT: Programs 2 and 3 (training / TFLite export) work for any plant folder, so reuse them with
  %run program2_train_mobilenetv4.py --plants apple corn grape pepper
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import subprocess
import sys

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PLANTDOC_GIT = "https://github.com/pratikkayal/PlantDoc-Dataset"
PLANTVILLAGE_GIT = "https://github.com/spMohanty/PlantVillage-Dataset"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def git_clone(url: str, dst: Path):
    if dst.exists() and any(dst.iterdir()):
        print(f"Using existing {dst}")
        return
    print(f"Cloning {url} (this can take a few minutes)...")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)


def download_to_browser(path: Path):
    """Trigger a browser download (Colab only)."""
    if in_colab():
        from google.colab import files
        files.download(str(path))
    else:
        print(f"(not in Colab) file saved at: {path}")

# canonical_name: (PlantVillage folder, PlantDoc folder)
# Matching ignores case/punctuation, so small naming differences are fine.
# Edit this if your downloaded copies use different folder names.
PLANT_CONFIG = {
    "apple": {
        "apple_scab":       ("Apple___Apple_scab", "Apple Scab Leaf"),
        "cedar_apple_rust": ("Apple___Cedar_apple_rust", "Apple rust leaf"),
        "healthy":          ("Apple___healthy", "Apple leaf"),
        # PlantVillage "Apple___Black_rot" has no PlantDoc partner -> thrown out
    },
    "corn": {
        "gray_leaf_spot":       ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
        "common_rust":          ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
        "northern_leaf_blight": ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
        # PlantVillage "Corn_(maize)___healthy" has no PlantDoc partner -> thrown out
    },
    "grape": {
        "black_rot": ("Grape___Black_rot", "grape leaf black rot"),
        "healthy":   ("Grape___healthy", "grape leaf"),
        # PlantVillage Esca + Leaf blight have no PlantDoc partner -> thrown out
    },
    "pepper": {  # bell pepper
        "bacterial_spot": ("Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot"),
        "healthy":        ("Pepper,_bell___healthy", "Bell_pepper leaf"),
    },
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def has_images(d: Path) -> bool:
    return any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())


def list_images(d: Path):
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def class_dirs(root: Path):
    """Map normalized folder name -> Path for every folder that directly holds images."""
    if root is None or not root.exists():
        return {}
    return {norm(d.name): d for d in root.rglob("*") if d.is_dir() and has_images(d)}


def find_split(root: Path, name: str):
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() == name:
            return d
    return None


def clean_label(folder_name: str, plant: str) -> str:
    s = folder_name.lower().replace(plant, " ")
    s = re.sub(r"\bleaf\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or norm(folder_name)


def copy_files(files, dst: Path, prefix: str):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f"{prefix}_{f.name}")


def build_plant(plant, pv_all, pd_train_all, pd_test_all, out_root, args):
    keyword = norm(plant)
    pd_train = {k: v for k, v in pd_train_all.items() if keyword in k}
    pd_test = {k: v for k, v in pd_test_all.items() if keyword in k}
    pv = {k: v for k, v in pv_all.items() if k.startswith(keyword)}

    if not pd_train:
        print(f"[{plant}] not found in PlantDoc -> skipped.")
        return None

    cfg = PLANT_CONFIG.get(plant, {})
    # canonical label for every PlantDoc class; and which of them have a PV partner
    label_of, pv_partner = {}, {}
    for canon, (pv_name, pd_name) in cfg.items():
        if norm(pd_name) in pd_train:
            label_of[norm(pd_name)] = canon
            if norm(pv_name) in pv:
                pv_partner[norm(pd_name)] = pv[norm(pv_name)]

    mode = "merged" if pv_partner else "plantdoc_only"
    for k, d in pd_train.items():  # PlantDoc-only extras are always kept
        label_of.setdefault(k, clean_label(d.name, plant))
    if len(label_of) < 2:
        print(f"[{plant}] only {len(label_of)} class left after applying the rules -> cannot train a classifier, skipped.")
        return None

    used_pv = {norm(p.name) for p in pv_partner.values()}
    dropped_pv = [pv[k].name for k in pv if k not in used_pv] if mode == "merged" else []

    out_dir = out_root / plant
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rng = random.Random(args.seed)
    report = {"plant": plant, "mode": mode, "classes": {}, "dropped_plantvillage": dropped_pv}

    for k, pd_dir in sorted(pd_train.items(), key=lambda kv: label_of[kv[0]]):
        label = label_of[k]
        train_pool = [("pd", f) for f in list_images(pd_dir)]
        if k in pv_partner:
            pv_imgs = list_images(pv_partner[k])
            rng.shuffle(pv_imgs)
            train_pool += [("pv", f) for f in pv_imgs[: args.max_pv_per_class]]

        if k in pd_test:
            test_files = [("pd", f) for f in list_images(pd_test[k])]
        else:  # no official test split for this class -> carve from PlantDoc train
            pd_only = [x for x in train_pool if x[0] == "pd"]
            rng.shuffle(pd_only)
            n = max(1, int(len(pd_only) * 0.15))
            test_files = pd_only[:n]
            held = {f for _, f in test_files}
            train_pool = [x for x in train_pool if x[1] not in held]

        rng.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * args.val_frac))
        val_files, train_files = train_pool[:n_val], train_pool[n_val:]

        for split, items in (("train", train_files), ("val", val_files), ("test", test_files)):
            for src in ("pd", "pv"):
                copy_files([f for s, f in items if s == src], out_dir / split / label, src)

        report["classes"][label] = {
            "plantdoc_folder": pd_dir.name,
            "plantvillage_folder": pv_partner[k].name if k in pv_partner else None,
            "train": len(train_files), "val": len(val_files), "test": len(test_files),
        }

    report["class_names"] = sorted(report["classes"])
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n[{plant}] mode = {mode}")
    for c, r in sorted(report["classes"].items()):
        src = "PD+PV" if r["plantvillage_folder"] else "PD only"
        print(f"   {c:26s} {src:8s} train={r['train']:5d} val={r['val']:4d} test={r['test']:4d}")
    if dropped_pv:
        print(f"   dropped PlantVillage-only classes: {dropped_pv}")
    return report


def main():
    base = Path("/content") if in_colab() else Path(".")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantvillage", type=Path, default=None)
    ap.add_argument("--plantdoc", type=Path, default=None)
    ap.add_argument("--out", default=base / "plant_datasets", type=Path)
    ap.add_argument("--plants", nargs="+", default=["apple", "corn", "grape", "pepper"])
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--max_pv_per_class", type=int, default=1000,
                    help="cap PlantVillage images per class so lab photos don't swamp PlantDoc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_download", action="store_true", help="skip the browser download")
    args, _ = ap.parse_known_args()  # tolerant of Jupyter/Colab's extra argv

    # Get the raw datasets (auto-clone if no path given)
    if args.plantdoc is None:
        args.plantdoc = base / "PlantDoc-Dataset"
        git_clone(PLANTDOC_GIT, args.plantdoc)
    if args.plantvillage is None:
        args.plantvillage = base / "PlantVillage-Dataset"
        git_clone(PLANTVILLAGE_GIT, args.plantvillage)
    # PlantVillage repo has color/grayscale/segmented copies: use colour only
    for cand in (args.plantvillage / "raw" / "color", args.plantvillage / "color"):
        if cand.exists():
            args.plantvillage = cand
            break

    pv_all = class_dirs(args.plantvillage)
    pd_train_root = find_split(args.plantdoc, "train") or args.plantdoc
    pd_test_root = find_split(args.plantdoc, "test")
    pd_train_all, pd_test_all = class_dirs(pd_train_root), class_dirs(pd_test_root)

    print(f"PlantVillage classes found: {len(pv_all)} | PlantDoc train classes: {len(pd_train_all)}")
    args.out.mkdir(parents=True, exist_ok=True)

    reports = [r for p in args.plants
               if (r := build_plant(p.lower(), pv_all, pd_train_all, pd_test_all, args.out, args))]
    print(f"\nDone. Built {len(reports)} plant dataset(s) in {args.out.resolve()}")

    # Zip each plant and send it to the browser
    for r in reports:
        zip_base = args.out / f"{r['plant']}_dataset"
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", args.out / r["plant"]))
        print(f"Zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        if not args.no_download:
            download_to_browser(zip_path)


if __name__ == "__main__":
    main()
