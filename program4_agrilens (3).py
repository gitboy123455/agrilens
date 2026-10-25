#!/usr/bin/env python3
"""
Program 4 (v2): AgriLens - plant leaf diagnosis kiosk for Raspberry Pi 4/5 (NOT for Colab).

  * Full-screen touch UI: live webcam preview, Capture / Diagnose / Clear buttons
  * "Choose Plant": the farmer taps a plant name (Tomato, Potato, ...); every plant has its own
    .tflite model, all listed in plants.json (new models dropped into /models are added automatically)
  * Capture -> photo quality check (blurry / dark / too bright / no leaf) -> if the photo is bad it
    STOPS and tells the farmer how to retake it, otherwise it diagnoses automatically
  * Treatment advice comes from disease_suggestions.json (editable)
  * Starts at boot. Can only be closed by an administrator, in one of two ways (both need the password):
      - the secret button in the top-right corner: tap it 7 times quickly, then enter the password
      - the keyboard combo Ctrl + Alt + Shift + Q (needs a USB keyboard), then enter the password
        (the password can be typed on the keyboard or on the on-screen keys)

ONE-TIME SETUP ON THE PI:
  1. Put this file in a folder, e.g. /home/pi/agrilens/
  2. Put each plant's files in /home/pi/agrilens/models/ :
         tomato_mobilenetv4.tflite + tomato_labels.txt, potato_mobilenetv4.tflite + potato_labels.txt ...
  3. python3 program4_agrilens.py --install     (run it AGAIN after updating this file:
         it rewrites the start-at-boot entry; it won't ask for the password a second time)
  4. sudo raspi-config -> System Options -> Boot / Auto Login -> Desktop Autologin
     (also Display Options -> Screen Blanking -> Off)
  5. sudo reboot

OTHER COMMANDS:
  python3 program4_agrilens.py                 run now (full screen)
  python3 program4_agrilens.py --windowed      run in a normal window (testing over VNC)
  python3 program4_agrilens.py --set-password  change the exit password
  python3 program4_agrilens.py --uninstall     remove start-at-boot
  Options: --camera 1 (camera index)   --hide-cursor (only if you use a touchscreen and no mouse)

FILES THIS APP CREATES NEXT TO ITSELF:
  plants.json              plant names shown to farmers -> model files (edit the "name" to rename)
  disease_suggestions.json advice texts
  config.json              password hash, last plant, thresholds (min_confidence, quality{...})
  captures/                every photo taken        diagnosis_log.csv   results + quality numbers
"""
import argparse
import csv
import getpass
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CAPTURES_DIR = APP_DIR / "captures"
CONFIG_PATH = APP_DIR / "config.json"
SUGGESTIONS_PATH = APP_DIR / "disease_suggestions.json"
PLANTS_PATH = APP_DIR / "plants.json"
LOG_PATH = APP_DIR / "diagnosis_log.csv"
LOGO_PATH = APP_DIR / "logo.png"  # optional: shown in the header if present
AUTOSTART_PATH = Path.home() / ".config" / "autostart" / "agrilens.desktop"
DEFAULT_PASSWORD = "agrilens"  # only used if --install / --set-password was never run

# ============================== SETTINGS (edit here) ==============================
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720      # capture size (HP w200 max is 1280x720; a 1080p camera: 1920, 1080)
KEYBOARD_EXIT_COMBO = "<Control-Alt-Shift-Q>"  # admin keyboard combo that stops the app
KEYBOARD_COMBO_NEEDS_PASSWORD = True         # True: combo opens the password box. False: combo quits at once
# ===================================================================================

# Photo-quality thresholds (starting values; tune them in config.json -> "quality")
QUALITY_DEFAULTS = {
    "enabled": True,
    "min_sharpness": 25.0,        # below = blurry
    "min_brightness": 50.0,       # below = too dark   (0-255)
    "max_brightness": 220.0,      # above = too bright (0-255)
    "min_leaf_fraction": 0.12,    # share of leaf-coloured pixels in the centre of the photo
}

REQUIRED_PACKAGES = {  # import name -> pip name
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "ai_edge_litert": "ai-edge-litert",
}
APT_HELP = ("Run these once, then start the script again:\n"
            "  sudo apt update\n"
            "  sudo apt install -y python3-pip python3-venv python3-tk python3-pil.imagetk")


def _have(mod):
    return importlib.util.find_spec(mod) is not None


def _missing():
    miss = [pip for mod, pip in REQUIRED_PACKAGES.items() if not _have(mod)]
    if "ai-edge-litert" in miss and _have("tflite_runtime"):  # older runtime also works
        miss.remove("ai-edge-litert")
    return miss


def _pip(python, pkgs):
    return subprocess.run([python, "-m", "pip", "install", "-q", *pkgs]).returncode == 0


def install_dependencies():
    """System command: make sure every dependency is available.

    Plain `pip install` is blocked on Raspberry Pi OS Bookworm ('externally-managed-environment'),
    so if it fails this builds a private virtual environment (.venv, which can still see apt's
    python3-tk) next to the script and restarts the script inside it.
    """
    if not _have("tkinter"):
        sys.exit("Tkinter is missing.\n" + APT_HELP)
    missing = _missing()
    if not missing:
        return
    print(f"Installing missing dependencies: {missing}")
    in_venv = sys.prefix != sys.base_prefix
    if _pip(sys.executable, missing):
        if not _missing():
            return
    if not in_venv:
        venv = APP_DIR / ".venv"
        vpy = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
        if not vpy.exists():
            print("Creating a private Python environment (.venv) ...")
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        if vpy.exists():
            if _pip(str(vpy), missing) or not _missing_in(str(vpy)):
                print("Restarting inside the .venv environment ...")
                os.execv(str(vpy), [str(vpy), *sys.argv])
            if _pip(str(vpy), [p for p in missing if p != "ai-edge-litert"] + ["tflite-runtime"]):
                print("ai-edge-litert is not available for this Python; using tflite-runtime instead.")
                os.execv(str(vpy), [str(vpy), *sys.argv])
    sys.exit(f"Could not install {_missing()}.\n" + APT_HELP)


def _missing_in(python):
    """Which required imports are still missing when run with a given interpreter."""
    code = ("import importlib.util as u;"
            + "print([m for m in %r if u.find_spec(m) is None])" % list(REQUIRED_PACKAGES))
    out = subprocess.run([python, "-c", code], capture_output=True, text=True).stdout.strip()
    import ast
    return ast.literal_eval(out) if out.startswith("[") else list(REQUIRED_PACKAGES)


install_dependencies()

DEFAULT_SUGGESTIONS = {
    "_meta": {
        "disclaimer": "General guidance only. Ask your local agriculture officer before using any chemical, and follow the product label."
    },
    "_default": {
        "display_name": "Unrecognised condition",
        "cause": "No advice is stored for this result yet.",
        "suggestions": [
            "Remove badly affected leaves and keep them away from healthy plants.",
            "Avoid wetting the leaves when watering; water at the base in the morning.",
            "Show the plant to your local agriculture officer for confirmation.",
            "You can add advice for this disease in disease_suggestions.json.",
        ],
    },
    "tomato": {
        "early_blight": {
            "display_name": "Early Blight",
            "cause": "Fungal disease (Alternaria solani), common in warm, humid weather.",
            "suggestions": [
                "Remove and destroy the affected lower leaves.",
                "Water at the base in the morning; avoid wetting the leaves.",
                "Mulch around plants so soil does not splash onto leaves.",
                "Stake and prune plants for good air flow.",
                "Rotate crops; do not plant tomato, potato or brinjal in the same soil year after year.",
                "If it keeps spreading, use a fungicide recommended by your local agriculture office.",
            ],
        },
        "late_blight": {
            "display_name": "Late Blight",
            "cause": "Water-mould disease (Phytophthora infestans); spreads very fast in cool, wet weather.",
            "suggestions": [
                "Act quickly: remove and destroy (do not compost) infected plants or leaves.",
                "Keep leaves dry; avoid overhead watering and improve spacing.",
                "Check nearby tomato and potato plants daily.",
                "Healthy plants nearby may need protective fungicide: ask your local agriculture officer.",
            ],
        },
        "septoria_leaf_spot": {
            "display_name": "Septoria Leaf Spot",
            "cause": "Fungal disease (Septoria lycopersici), spread by water splash.",
            "suggestions": [
                "Remove spotted lower leaves and destroy them.",
                "Avoid overhead watering and working among wet plants.",
                "Mulch the soil and keep plants staked off the ground.",
                "Rotate crops and clean tools; ask your local agriculture office about fungicide if severe.",
            ],
        },
        "bacterial_spot": {
            "display_name": "Bacterial Spot",
            "cause": "Bacterial disease (Xanthomonas), spread by water splash and infected seed.",
            "suggestions": [
                "Remove infected leaves; do not handle plants when they are wet.",
                "Use clean, disease-free seed and seedlings.",
                "Avoid overhead watering; keep plants well spaced.",
                "Copper-based sprays are often used early on; confirm with your local agriculture officer.",
            ],
        },
        "leaf_mold": {
            "display_name": "Leaf Mold",
            "cause": "Fungal disease (Passalora fulva), favoured by high humidity and poor ventilation.",
            "suggestions": [
                "Improve air flow: open vents or space and prune plants.",
                "Reduce humidity and keep leaves dry.",
                "Remove and destroy infected leaves.",
                "Choose resistant varieties next season.",
            ],
        },
        "mosaic_virus": {
            "display_name": "Mosaic Virus",
            "cause": "Virus spread by hands, tools and seed. It cannot be cured.",
            "suggestions": [
                "Remove and destroy infected plants; do not compost them.",
                "Wash hands with soap and clean tools before touching healthy plants.",
                "Do not handle tobacco before working with tomato plants.",
                "Use certified seed and resistant varieties; control weeds.",
            ],
        },
        "yellow_leaf_curl_virus": {
            "display_name": "Yellow Leaf Curl Virus",
            "cause": "Virus spread by whiteflies. It cannot be cured.",
            "suggestions": [
                "Remove and destroy infected plants early.",
                "Control whiteflies: yellow sticky traps, insect-proof nets in the nursery.",
                "Neem-based sprays can help reduce whiteflies; ask your local agriculture office for options.",
                "Remove weeds and use resistant varieties.",
            ],
        },
        "spider_mites": {
            "display_name": "Spider Mites",
            "cause": "Tiny sap-sucking pests (two-spotted spider mite), worse in hot, dry weather.",
            "suggestions": [
                "Spray the underside of leaves with a strong jet of water.",
                "Remove heavily infested leaves.",
                "Keep plants well watered; drought-stressed plants suffer more.",
                "Neem oil or insecticidal soap can help; avoid harsh sprays that kill helpful insects.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Keep watering at the base and avoid wetting leaves.",
                "Check plants regularly, especially after rain.",
                "Keep the area weed-free and remove fallen leaves.",
            ],
        },
    },
    "potato": {
        "early_blight": {
            "display_name": "Early Blight",
            "cause": "Fungal disease (Alternaria solani), common when plants are stressed.",
            "suggestions": [
                "Remove and destroy badly affected leaves.",
                "Keep plants well fed and watered; stressed plants get it more.",
                "Rotate with non-potato crops.",
                "If it keeps spreading, ask your local agriculture office about a suitable fungicide.",
            ],
        },
        "late_blight": {
            "display_name": "Late Blight",
            "cause": "Water-mould disease (Phytophthora infestans); spreads very fast in cool, wet weather.",
            "suggestions": [
                "Act quickly: destroy infected plants (do not compost) to protect the field.",
                "Earth up (hill) the soil well so tubers are covered.",
                "Do not store tubers from badly infected plants.",
                "Ask your local agriculture office about protective fungicide for healthy plants.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Keep checking plants, especially in cool, wet weather.",
                "Water at the base and avoid wetting leaves.",
                "Use clean seed potatoes next season.",
            ],
        },
    },
    "mango": {
        "anthracnose": {
            "display_name": "Anthracnose",
            "cause": "Fungal disease (Colletotrichum sp.), favoured by wet, humid weather.",
            "suggestions": [
                "Remove affected leaves, twigs and fruit; destroy them.",
                "Prune the tree so air and sunlight reach the inside.",
                "Avoid overhead irrigation, especially at flowering.",
                "Ask your local agriculture office about a suitable fungicide and spray timing.",
            ],
        },
        "powdery_mildew": {
            "display_name": "Powdery Mildew",
            "cause": "Fungal disease (Oidium), white powder on flowers and young leaves.",
            "suggestions": [
                "Remove badly affected parts and prune for air flow.",
                "Check trees closely at flowering time.",
                "Ask your local agriculture office about a suitable fungicide and timing.",
            ],
        },
        "bacterial_canker": {
            "display_name": "Bacterial Canker",
            "cause": "Bacterial disease (Xanthomonas), enters through wounds and spreads in rain and wind.",
            "suggestions": [
                "Prune and destroy infected branches; clean tools afterwards.",
                "Avoid wounding the tree; use windbreaks if possible.",
                "Copper-based sprays are often used; confirm with your local agriculture officer.",
            ],
        },
        "die_back": {
            "display_name": "Die Back",
            "cause": "Fungal disease (Lasiodiplodia), causes twigs and branches to dry from the tip.",
            "suggestions": [
                "Cut dead branches back to healthy wood and destroy them.",
                "Seal large cuts with a protective paste.",
                "Avoid water stress and wounding; ask your local agriculture office about sprays.",
            ],
        },
        "gall_midge": {
            "display_name": "Gall Midge",
            "cause": "Insect pest; larvae make small galls on leaves and flower buds.",
            "suggestions": [
                "Remove and destroy galled leaves and shoots.",
                "Turn the soil under the tree to expose resting pupae.",
                "Ask your local agriculture office about control options.",
            ],
        },
        "sooty_mould": {
            "display_name": "Sooty Mould",
            "cause": "Black fungus growing on honeydew left by sap-sucking insects.",
            "suggestions": [
                "Control the insects (hoppers, scales, mealybugs) that cause it.",
                "Prune crowded branches for air flow.",
                "Wash leaves if possible; the mould fades once insects are controlled.",
            ],
        },
        "cutting_weevil": {
            "display_name": "Cutting Weevil",
            "cause": "Insect pest (leaf-cutting weevil) that cuts young leaves.",
            "suggestions": [
                "Collect and destroy cut leaves and any weevils you find.",
                "Check young flushes regularly.",
                "Ask your local agriculture office about control options.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Prune for air flow and remove dead twigs.",
                "Inspect new growth and flowers regularly.",
            ],
        },
    },
}


NEW_PLANT_SUGGESTIONS = {
    "apple": {
        "apple_scab": {
            "display_name": "Apple Scab",
            "cause": "Fungal disease (Venturia inaequalis), worst in cool, wet spring weather.",
            "suggestions": [
                "Rake up and destroy fallen leaves in autumn; they carry the fungus over winter.",
                "Prune the tree so air and sun reach the inside of the canopy.",
                "Avoid overhead watering and wetting the leaves.",
                "Ask your local agriculture office about fungicide and the right spray timing.",
            ],
        },
        "cedar_apple_rust": {
            "display_name": "Cedar Apple Rust",
            "cause": "Fungal disease (Gymnosporangium) that also lives on nearby juniper / cedar trees.",
            "suggestions": [
                "Remove and destroy badly affected leaves.",
                "If juniper or cedar trees grow close by, remove their galls or the trees if possible.",
                "Choose resistant apple varieties when planting new trees.",
                "Ask your local agriculture office about fungicide timing in spring.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Prune for air flow and remove fallen leaves.",
                "Check leaves regularly, especially in wet weather.",
            ],
        },
    },
    "corn": {
        "gray_leaf_spot": {
            "display_name": "Gray Leaf Spot",
            "cause": "Fungal disease (Cercospora), favoured by warm, humid weather and old crop residue.",
            "suggestions": [
                "Rotate with a non-maize crop and plough in old maize stalks and leaves.",
                "Avoid planting too densely so air can move between plants.",
                "Choose tolerant varieties next season.",
                "If it is spreading early, ask your local agriculture office about fungicide.",
            ],
        },
        "common_rust": {
            "display_name": "Common Rust",
            "cause": "Fungal disease (Puccinia sorghi), spreads on wind in cool, moist weather.",
            "suggestions": [
                "Check plants regularly; mild rust late in the season usually needs no treatment.",
                "Choose resistant varieties next season.",
                "If heavy rust appears early, ask your local agriculture office about fungicide.",
            ],
        },
        "northern_leaf_blight": {
            "display_name": "Northern Leaf Blight",
            "cause": "Fungal disease (Exserohilum turcicum), long grey-green lesions in humid, moderate weather.",
            "suggestions": [
                "Rotate crops and plough in old maize residue after harvest.",
                "Choose resistant varieties next season.",
                "Avoid very dense planting.",
                "If it appears before flowering, ask your local agriculture office about fungicide.",
            ],
        },
    },
    "grape": {
        "black_rot": {
            "display_name": "Black Rot",
            "cause": "Fungal disease (Guignardia bidwellii), spread by rain splash in warm, wet weather.",
            "suggestions": [
                "Remove and destroy infected leaves and shrivelled (mummified) berries.",
                "Prune so the vines dry quickly after rain.",
                "Keep weeds down under the vines for better air flow.",
                "Ask your local agriculture office about protective fungicide and timing.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Keep vines pruned and open to air and sun.",
                "Check leaves and berries regularly, especially after rain.",
            ],
        },
    },
    "pepper": {
        "bacterial_spot": {
            "display_name": "Bacterial Spot",
            "cause": "Bacterial disease (Xanthomonas), spread by water splash, tools and infected seed.",
            "suggestions": [
                "Remove infected leaves; do not work among wet plants.",
                "Use clean, disease-free seed and seedlings.",
                "Water at the base; avoid overhead watering.",
                "Rotate away from pepper and tomato for a few seasons.",
                "Copper-based sprays are often used early; confirm with your local agriculture officer.",
            ],
        },
        "healthy": {
            "display_name": "Healthy Leaf",
            "cause": "No disease detected.",
            "suggestions": [
                "Keep watering at the base and avoid wetting leaves.",
                "Check plants regularly and remove fallen leaves.",
            ],
        },
    },
}
DEFAULT_SUGGESTIONS.update(NEW_PLANT_SUGGESTIONS)


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def ensure_suggestions_file():
    """Create the advice file, or ADD plants/diseases that are new in this version (never overwrites edits)."""
    if not SUGGESTIONS_PATH.exists():
        SUGGESTIONS_PATH.write_text(json.dumps(DEFAULT_SUGGESTIONS, indent=2, ensure_ascii=False))
        print(f"Created {SUGGESTIONS_PATH}")
        return
    try:
        raw = json.loads(SUGGESTIONS_PATH.read_text())
        changed = False
        for plant, classes in DEFAULT_SUGGESTIONS.items():
            if plant not in raw:
                raw[plant], changed = classes, True
            elif not plant.startswith("_") and isinstance(raw[plant], dict):
                for cname, entry in classes.items():
                    if cname not in raw[plant]:
                        raw[plant][cname], changed = entry, True
        if changed:
            SUGGESTIONS_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
            print(f"Added new advice entries to {SUGGESTIONS_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not update {SUGGESTIONS_PATH}: {e}")


def load_suggestions():
    ensure_suggestions_file()
    try:
        raw = json.loads(SUGGESTIONS_PATH.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"Could not read suggestions JSON ({e}); using built-in defaults.")
        raw = DEFAULT_SUGGESTIONS
    db = {"_default": raw.get("_default", DEFAULT_SUGGESTIONS["_default"]),
          "_meta": raw.get("_meta", {})}
    for plant, classes in raw.items():
        if not plant.startswith("_") and isinstance(classes, dict):
            db[key(plant)] = {key(c): v for c, v in classes.items()}
    return db


def find_advice(db, plant, label):
    k = key(label)
    if k in db.get(key(plant), {}):
        return db[key(plant)][k]
    for p, classes in db.items():  # class not under this plant: search other plants
        if not p.startswith("_") and k in classes and k != "healthy":
            return classes[k]
    return db["_default"]


# ----------------------------------------------------------------------------
# Config + password
# ----------------------------------------------------------------------------
def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def hash_password(pw, salt=None, iterations=200_000):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iterations)
    return {"salt": salt.hex(), "hash": h.hex(), "iterations": iterations}


def check_password(pw, cfg):
    rec = cfg.get("password")
    if not rec:
        return hmac.compare_digest(pw.encode(), DEFAULT_PASSWORD.encode())
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(rec["salt"]), rec["iterations"])
    return hmac.compare_digest(h.hex(), rec["hash"])


def prompt_new_password():
    while True:
        p1 = getpass.getpass("Choose an exit password (min 4 characters): ")
        if len(p1) < 4:
            print("Too short.")
            continue
        if p1 != getpass.getpass("Repeat password: "):
            print("Passwords do not match.")
            continue
        cfg = load_config()
        cfg["password"] = hash_password(p1)
        save_config(cfg)
        print("Password saved.")
        return


# ----------------------------------------------------------------------------
# Plants registry (plants.json): plant name shown to farmers -> model + labels files
# ----------------------------------------------------------------------------
def resolve_path(s) -> Path:
    p = Path(s)
    return p if p.is_absolute() else APP_DIR / p


def store_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(APP_DIR))
    except ValueError:
        return str(p.resolve())


def find_labels(model_path: Path):
    stem, d = model_path.stem, model_path.parent
    plant = stem.split("_")[0]
    for c in (d / f"{stem.replace('_mobilenetv4', '')}_labels.txt", d / f"{stem}_labels.txt",
              d / f"{stem}.txt", d / f"{plant}_labels.txt", d / "labels.txt"):
        if c.exists():
            return c
    return None


def plant_key_from(model_path: Path, labels_path: Path) -> str:
    s = labels_path.stem
    return s[: -len("_labels")] if s.endswith("_labels") else model_path.stem.split("_")[0]


def scan_models():
    """All .tflite files (with a labels file) in the models folder and on USB sticks."""
    out, seen = [], set()
    for base in (MODELS_DIR, Path("/media")):
        if not base.exists():
            continue
        depth0 = len(base.parts)
        for dirpath, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if len(Path(dirpath).parts) - depth0 >= 3:
                dirs[:] = []
            for f in sorted(files):
                if f.lower().endswith(".tflite"):
                    m = Path(dirpath) / f
                    lab = find_labels(m)
                    if lab and m.resolve() not in seen:
                        seen.add(m.resolve())
                        out.append((m, lab))
    return out


def sync_plants():
    """Load plants.json and add any newly found model. Returns the list of plant entries."""
    try:
        data = json.loads(PLANTS_PATH.read_text())
        entries = data["plants"] if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001
        entries = []
    known = {resolve_path(e.get("model", "")).resolve() for e in entries if e.get("model")}
    names = {str(e.get("name", "")).lower() for e in entries}
    changed = not PLANTS_PATH.exists()
    for model, labels in scan_models():
        if model.resolve() in known:
            continue
        key_ = plant_key_from(model, labels)
        name = key_.replace("_", " ").title()
        while name.lower() in names:
            name += " 2"
        names.add(name.lower())
        entries.append({"name": name, "key": key_, "model": store_path(model), "labels": store_path(labels)})
        changed = True
    if changed:
        PLANTS_PATH.write_text(json.dumps({
            "_help": "'name' is what farmers see on the button - rename it freely (any language). "
                     "'key' picks the advice in disease_suggestions.json.",
            "plants": entries}, indent=2, ensure_ascii=False))
    return entries


def entry_available(e) -> bool:
    return bool(e.get("model")) and resolve_path(e["model"]).exists() and resolve_path(e.get("labels", "")).exists()


# ----------------------------------------------------------------------------
# Heavy libraries are loaded AFTER the window is on screen (faster visible start-up)
# ----------------------------------------------------------------------------
cv2 = np = Image = ImageTk = None


def load_heavy():
    global cv2, np, Image, ImageTk
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Image
    try:
        from PIL import ImageTk as _ImageTk
    except ImportError:
        sys.exit("Run: sudo apt install python3-pil.imagetk")
    cv2, np, Image, ImageTk = _cv2, _np, _Image, _ImageTk


class Classifier:
    def __init__(self, model_path: Path, labels_path: Path, threads=4):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
        self.labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        self.interp = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shape = [int(v) for v in self.inp["shape"]]
        if shape[1] == 3:
            self.nchw, self.size = True, shape[2]
        elif shape[3] == 3:
            self.nchw, self.size = False, shape[1]
        else:
            raise ValueError(f"Unexpected model input shape {shape}")
        n_out = int(self.out["shape"][-1])
        if n_out != len(self.labels):
            raise ValueError(f"Model has {n_out} outputs but labels file has {len(self.labels)} lines.")

    def predict(self, bgr, top=3):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = int(self.size * 1.14) / min(h, w)  # same resize + centre-crop as validation
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        y0, x0 = (h - self.size) // 2, (w - self.size) // 2
        rgb = rgb[y0:y0 + self.size, x0:x0 + self.size]
        if self.inp["dtype"] == np.uint8:
            x = rgb.astype(np.uint8)
        else:
            x = rgb.astype(np.float32) / 255.0
        if self.nchw:
            x = x.transpose(2, 0, 1)
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()
        p = self.interp.get_tensor(self.out["index"])[0].astype(np.float64)
        if p.min() < 0 or abs(p.sum() - 1) > 1e-2:  # logits -> probabilities
            e = np.exp(p - p.max())
            p = e / e.sum()
        order = np.argsort(p)[::-1][:top]
        return [(self.labels[i], float(p[i])) for i in order]


# ----------------------------------------------------------------------------
# Photo quality check
# ----------------------------------------------------------------------------
def check_quality(bgr, q):
    """Returns (ok, farmer_message or None, metrics dict). Looks at the centre square (what the AI sees)."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    crop = bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    small = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "leaf_fraction": float(((hsv[..., 0] >= 10) & (hsv[..., 0] <= 100) &
                                (hsv[..., 1] >= 35) & (hsv[..., 2] >= 35)).mean()),
    }
    if not q.get("enabled", True):
        return True, None, m
    if m["brightness"] < q["min_brightness"]:
        return False, "The photo is too dark. Move to daylight or switch on a light, then tap Capture again.", m
    if m["brightness"] > q["max_brightness"]:
        return False, "The photo is too bright (glare). Move out of direct sunlight or shade the leaf, then tap Capture again.", m
    if m["leaf_fraction"] < q["min_leaf_fraction"]:
        return False, "I cannot see a leaf clearly. Hold ONE leaf close so it fills the picture, then tap Capture again.", m
    if m["sharpness"] < q["min_sharpness"]:
        return False, "The photo is blurry. Hold the leaf and the camera steady, then tap Capture again.", m
    return True, None, m


# ----------------------------------------------------------------------------
# Camera runs in its own thread so the touch screen never waits for the webcam
# ----------------------------------------------------------------------------
class CameraStream(threading.Thread):
    def __init__(self, index):
        super().__init__(daemon=True)
        self.index = index
        self.lock = threading.Lock()
        self.frame, self.stamp, self.opened = None, 0, False
        self.halt = threading.Event()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2) if sys.platform.startswith("linux") \
            else cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        return None

    def run(self):
        cap, fails = None, 0
        while not self.halt.is_set():
            if cap is None:
                cap = self._open()
                self.opened = cap is not None
                if cap is None:
                    time.sleep(2)
                    continue
            ok, frame = cap.read()
            if ok:
                fails = 0
                with self.lock:
                    self.frame, self.stamp = frame, self.stamp + 1
            else:
                fails += 1
                time.sleep(0.05)
                if fails > 40:
                    cap.release()
                    cap, self.opened = None, False
        if cap is not None:
            cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.stamp


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
import tkinter as tk  # noqa: E402

GREEN, GREEN_DK = "#2e7d32", "#1b5e20"
BLUE, GRAY = "#1565c0", "#607d8b"
BG, CARD, TXT = "#eef3ee", "#ffffff", "#1b1b1b"
RED, ORANGE = "#c62828", "#e65100"
SECRET_TAPS, TAP_WINDOW_S = 7, 2.5
_K = [1.0]
HINT = "Choose your plant, hold ONE leaf in front of the camera so it fills the picture, then tap Capture. The result appears automatically."


def PX(n):
    """Font size in PIXELS (negative = pixels in Tk), so the look does not depend on the system DPI."""
    return -max(9, int(n * _K[0] * 1.3))


def make_root(args):
    os.environ.setdefault("XCURSOR_SIZE", "24")  # sane pointer size when started before the desktop theme
    deadline = time.time() + 120
    while True:  # at boot the display may not be ready yet: keep trying instead of crashing
        try:
            root = tk.Tk()
            break
        except tk.TclError:
            if time.time() > deadline:
                raise
            time.sleep(0.5)
    root.title("AgriLens")
    root.configure(bg=GREEN)
    if args.windowed:
        root.geometry("800x480")
    else:
        root.attributes("-fullscreen", True)
    tk.Label(root, text="AgriLens\nStarting, please wait...", bg=GREEN, fg="white",
             font=("Helvetica", -34, "bold")).place(relx=0.5, rely=0.5, anchor="center")
    root.update()
    return root


class AgriLensApp:
    def __init__(self, root, cfg, args):
        self.root, self.cfg, self.args = root, cfg, args
        self.fullscreen = not args.windowed
        self.sw, self.sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.k = max(0.7, self.sh / 480)
        _K[0] = self.k
        self.db = load_suggestions()
        self.clf, self.plant_key, self.plant_label, self.plant_model = None, "", "", ""
        self.live_stamp, self.frozen, self.photo_path, self.tkimg = -1, None, "", None
        self.cam_state, self.overlay, self.t0 = None, None, time.time()
        self.tap_count, self.last_tap, self.fails, self.locked_until = 0, 0.0, 0, 0.0
        self.pw_active, self.pw_var, self.pw_submit = False, None, None

        for w in root.winfo_children():  # remove the splash text
            w.destroy()
        root.configure(bg=BG)
        if self.fullscreen:
            try:
                root.attributes("-topmost", True)
            except tk.TclError:
                pass
            for ms in (800, 2500, 6000):  # panels that appear late must not cover the app
                root.after(ms, self.reassert)
        root.protocol("WM_DELETE_WINDOW", lambda: None)  # the window can't be closed normally
        if args.hide_cursor:
            root.config(cursor="none")

        self.build_ui()
        self.cam = CameraStream(args.camera)
        self.cam.start()
        self.update_status()
        self.tick()
        root.after(200, self.restore_last_plant)

        # keyboard: admin combo + typing into the password box
        root.bind("<Key>", self.on_key)
        combo = KEYBOARD_EXIT_COMBO
        for c in {combo, combo[:-2] + combo[-2].lower() + ">", combo[:-2] + combo[-2].upper() + ">"}:
            try:
                root.bind_all(c, self.on_exit_combo)
            except tk.TclError:
                print(f"WARNING: invalid KEYBOARD_EXIT_COMBO {c!r}")
        root.focus_force()

    def on_exit_combo(self, event=None):
        if self.pw_active:
            return
        now = time.time()
        if now < self.locked_until:
            self.popup("Locked", f"Too many wrong attempts. Try again in {int(self.locked_until - now)} seconds.")
        elif KEYBOARD_COMBO_NEEDS_PASSWORD:
            self.ask_password()
        else:
            self.exit_app()

    def on_key(self, e):
        """Physical keyboard typing while the password box is open."""
        if not self.pw_active:
            return
        if e.keysym in ("Return", "KP_Enter"):
            self.pw_submit()
        elif e.keysym == "Escape":
            self.close_overlay()
        elif e.keysym == "BackSpace":
            self.pw_var.set(self.pw_var.get()[:-1])
        elif e.char and e.char.isprintable():
            self.pw_var.set(self.pw_var.get() + e.char)

    def reassert(self):
        try:
            self.root.attributes("-fullscreen", True)
            self.root.attributes("-topmost", True)
            if self.overlay is not None:
                self.overlay.lift()
        except tk.TclError:
            pass

    def fs(self, n):
        return max(8, int(n * self.k))

    # ---------- layout ----------
    def build_ui(self):
        r, fs = self.root, self.fs
        header = tk.Frame(r, bg=GREEN, height=fs(46))
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="AGRILENS  -  Plant Leaf Diagnosis", bg=GREEN, fg="white",
                 font=("Helvetica", PX(15), "bold")).pack(side="left", padx=fs(12))
        # secret button: small, unlabeled, top-right corner
        tk.Button(header, text="\u00b7", bg=GREEN_DK, fg=GREEN_DK, activebackground=GREEN,
                  activeforeground=GREEN, bd=0, highlightthickness=0, relief="flat",
                  font=("Helvetica", PX(10)), width=4, command=self.secret_tap).pack(side="right", fill="y")
        if LOGO_PATH.exists():
            try:
                logo = Image.open(LOGO_PATH)
                logo.thumbnail((fs(40), fs(40)))
                self.logo_img = ImageTk.PhotoImage(logo)
                tk.Label(header, image=self.logo_img, bg=GREEN).pack(side="right", padx=fs(6))
            except Exception:  # noqa: BLE001
                pass
        tk.Button(header, text="Choose Plant", bg="#43a047", fg="white", activebackground="#66bb6a",
                  activeforeground="white", font=("Helvetica", PX(12), "bold"), bd=0, padx=fs(14),
                  command=self.choose_plant).pack(side="right", padx=fs(8), pady=fs(5))

        self.status = tk.Label(r, text="", bg="#dfe8df", fg="#333", anchor="w",
                               font=("Helvetica", PX(9)), padx=fs(8))
        self.status.pack(side="bottom", fill="x")

        bar = tk.Frame(r, bg=BG)
        bar.pack(side="bottom", fill="x", padx=fs(10), pady=fs(8))
        for i, (txt, col, cmd) in enumerate([("Capture", BLUE, self.capture),
                                             ("Diagnose", GREEN, self.diagnose),
                                             ("Clear", GRAY, self.clear)]):
            bar.columnconfigure(i, weight=1, uniform="b")
            tk.Button(bar, text=txt, bg=col, fg="white", activebackground=col, activeforeground="white",
                      bd=0, font=("Helvetica", PX(16), "bold"), command=cmd, pady=fs(10)
                      ).grid(row=0, column=i, sticky="ew", padx=fs(5))

        body = tk.Frame(r, bg=BG)
        body.pack(side="top", fill="both", expand=True, padx=fs(10), pady=(fs(8), 0))
        body.columnconfigure(0, weight=1, uniform="c")
        body.columnconfigure(1, weight=1, uniform="c")
        body.rowconfigure(0, weight=1)

        self.photo_frame = tk.Frame(body, bg="black")
        self.photo_frame.grid(row=0, column=0, sticky="nsew", padx=(0, fs(6)))
        self.photo_frame.pack_propagate(False)
        self.photo_lbl = tk.Label(self.photo_frame, bg="black", fg="white", text="Starting camera...",
                                  font=("Helvetica", PX(12)))
        self.photo_lbl.place(relx=0.5, rely=0.5, anchor="center")

        info = tk.Frame(body, bg=CARD, padx=fs(10), pady=fs(6))
        info.grid(row=0, column=1, sticky="nsew")
        wrap = max(200, self.sw // 2 - fs(50))
        tk.Label(info, text="Diagnosis (Probable):", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", PX(10))).pack(fill="x")
        self.diag_lbl = tk.Label(info, text="-", bg=CARD, fg=TXT, anchor="w", justify="left",
                                 wraplength=wrap, font=("Helvetica", PX(20), "bold"))
        self.diag_lbl.pack(fill="x")
        self.conf_lbl = tk.Label(info, text="Confidence: -", bg=CARD, fg=TXT, anchor="w",
                                 wraplength=wrap, justify="left", font=("Helvetica", PX(12), "bold"))
        self.conf_lbl.pack(fill="x", pady=(fs(2), fs(4)))
        tk.Label(info, text="Possible Cause:", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", PX(10), "bold")).pack(fill="x")
        self.cause_lbl = tk.Label(info, text="-", bg=CARD, fg=TXT, anchor="w", justify="left",
                                  wraplength=wrap, font=("Helvetica", PX(11)))
        self.cause_lbl.pack(fill="x", pady=(0, fs(4)))
        tk.Label(info, text="Suggestion:", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", PX(10), "bold")).pack(fill="x")
        tf = tk.Frame(info, bg=CARD)
        tf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(tf, width=fs(26))
        sb.pack(side="right", fill="y")
        self.sugg = tk.Text(tf, bg=CARD, fg=TXT, bd=0, wrap="word", font=("Helvetica", PX(11)),
                            yscrollcommand=sb.set, state="disabled", cursor="arrow")
        self.sugg.pack(side="left", fill="both", expand=True)
        sb.config(command=self.sugg.yview)
        self.reset_result(HINT)

    def update_status(self):
        cam = "Camera OK" if self.cam.opened else "Camera not found" if self.cam_state is False else "Camera starting"
        plant = self.plant_label or "not chosen yet"
        self.status.config(text=f"Plant: {plant}   |   {cam}   |   {self.db['_meta'].get('disclaimer', '')}")

    def set_suggestions(self, items):
        self.sugg.config(state="normal")
        self.sugg.delete("1.0", "end")
        for it in items:
            self.sugg.insert("end", f"\u2022 {it}\n\n")
        self.sugg.config(state="disabled")

    def reset_result(self, hint="-"):
        self.diag_lbl.config(text="-", fg=TXT)
        self.conf_lbl.config(text="Confidence: -")
        self.cause_lbl.config(text="-")
        self.set_suggestions([hint] if hint != "-" else [])

    # ---------- camera view ----------
    def show(self, bgr):
        w, h = self.photo_frame.winfo_width(), self.photo_frame.winfo_height()
        if w < 20 or h < 20:
            return
        ih, iw = bgr.shape[:2]
        sc = min(w / iw, h / ih)
        small = cv2.resize(bgr, (max(1, int(iw * sc)), max(1, int(ih * sc))))
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.photo_lbl.config(image=self.tkimg, text="")

    def tick(self):
        if self.cam.opened != self.cam_state and (self.cam.opened or time.time() - self.t0 > 5):
            self.cam_state = self.cam.opened
            self.update_status()
            if not self.cam.opened:
                self.photo_lbl.config(image="", text="Camera not found.\nCheck the webcam cable.")
        if self.frozen is None:
            frame, stamp = self.cam.latest()
            if frame is not None and stamp != self.live_stamp:
                self.live_stamp = stamp
                self.show(frame)
        self.root.after(80, self.tick)

    # ---------- buttons ----------
    def capture(self):
        frame, _ = self.cam.latest()
        if frame is None:
            self.popup("Camera problem", "No camera picture yet. Check the webcam cable and try again.")
            return
        self.frozen = frame.copy()
        self.show(self.frozen)
        self.photo_path = ""
        try:
            CAPTURES_DIR.mkdir(exist_ok=True)
            self.photo_path = str(CAPTURES_DIR / time.strftime("leaf_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(self.photo_path, self.frozen)
        except Exception:  # noqa: BLE001
            pass
        self.reset_result("Checking the photo...")
        self.root.update_idletasks()
        self.diagnose()  # automatic

    def clear(self):
        self.frozen, self.live_stamp = None, -1
        self.reset_result(HINT)

    def quality_settings(self):
        return {**QUALITY_DEFAULTS, **self.cfg.get("quality", {})}

    def diagnose(self):
        if self.frozen is None:
            self.popup("No photo yet", "Tap Capture to take a photo of the leaf first.")
            return
        ok, problem, metrics = check_quality(self.frozen, self.quality_settings())
        if not ok:  # STOP: bad photo -> go back to the live picture and explain how to retake
            self.frozen, self.live_stamp = None, -1
            self.diag_lbl.config(text="Photo not good enough", fg=ORANGE)
            self.conf_lbl.config(text="Please take the photo again.")
            self.cause_lbl.config(text=problem)
            self.set_suggestions(["Use ONE leaf, close to the camera, so it fills the picture.",
                                  "Use daylight, but keep the leaf out of harsh direct sun.",
                                  "Hold still for a second, then tap Capture."])
            self.log_result("-", "BAD PHOTO", 0, metrics)
            return
        if self.clf is None:
            self.choose_plant("Please choose your plant first.")
            return
        self.diag_lbl.config(text="Analysing...", fg=TXT)
        self.root.update_idletasks()
        try:
            preds = self.clf.predict(self.frozen)
        except Exception as e:  # noqa: BLE001
            self.popup("Something went wrong", f"The leaf could not be checked. Please call your supervisor.\n\n({e})")
            return
        label, prob = preds[0]
        entry = find_advice(self.db, self.plant_key, label)
        name = entry.get("display_name") or label.replace("_", " ").title()
        self.log_result(self.plant_label, label, prob, metrics)
        if prob < self.cfg.get("min_confidence", 0.5):
            self.diag_lbl.config(text="Not sure", fg=ORANGE)
            self.conf_lbl.config(text=f"Confidence: {prob * 100:.0f}% (too low)")
            self.cause_lbl.config(text="The photo may be unclear, or this leaf does not match the plant you chose.")
            self.set_suggestions([
                "Check that you chose the right plant (tap Choose Plant).",
                "Retake the photo: one leaf, filling the picture, good daylight, plain background.",
                f"Closest guess: {name} ({prob * 100:.0f}%). Do not rely on it.",
            ])
            return
        healthy = key(label) == "healthy"
        self.diag_lbl.config(text=name, fg=GREEN if healthy else RED)
        self.conf_lbl.config(text=f"Confidence: {prob * 100:.0f}%")
        self.cause_lbl.config(text=entry.get("cause", "-"))
        self.set_suggestions(entry.get("suggestions", []))

    def log_result(self, plant, result, conf, metrics):
        try:
            new = not LOG_PATH.exists()
            with open(LOG_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "plant", "result", "confidence", "sharpness", "brightness",
                                "leaf_fraction", "photo"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), plant, result, f"{conf:.3f}",
                            f"{metrics['sharpness']:.1f}", f"{metrics['brightness']:.1f}",
                            f"{metrics['leaf_fraction']:.2f}", self.photo_path])
        except Exception:  # noqa: BLE001
            pass

    # ---------- plants ----------
    def restore_last_plant(self):
        last = self.cfg.get("last_plant")
        for e in sync_plants():
            if entry_available(e) and str(resolve_path(e["model"]).resolve()) == str(last):
                self.load_plant(e)
                break

    def load_plant(self, e):
        model, labels = resolve_path(e["model"]), resolve_path(e["labels"])
        try:
            self.clf = Classifier(model, labels)
        except Exception as ex:  # noqa: BLE001
            self.clf = None
            self.popup("This plant is not ready",
                       f"Please call your supervisor.\n\n({e.get('name')}: {ex})")
            return False
        self.plant_label = e.get("name", model.stem)
        self.plant_key = e.get("key") or plant_key_from(model, labels)
        self.cfg["last_plant"] = str(model.resolve())
        save_config(self.cfg)
        self.update_status()
        return True

    def pick_plant(self, e):
        self.close_overlay()
        if self.load_plant(e):
            if self.frozen is not None:
                self.diagnose()  # a photo is already waiting: diagnose it automatically
            else:
                self.reset_result(f"{self.plant_label} chosen. {HINT.split(', ', 1)[1]}")

    def choose_plant(self, note=None):
        f = self.fs
        plants = [e for e in sync_plants() if entry_available(e)]
        body = self.open_overlay("Which plant is it?")
        if note:
            tk.Label(body, text=note, bg=CARD, fg=ORANGE, font=("Helvetica", PX(13), "bold")).pack(pady=(0, f(4)))
        if not plants:
            tk.Label(body, text="No plants are installed yet.\nPlease ask your supervisor to add the plant files.",
                     bg=CARD, fg=TXT, font=("Helvetica", PX(15))).pack(expand=True)
        else:
            grid = tk.Frame(body, bg=CARD)
            grid.pack(expand=True, fill="both")
            cols = min(3, len(plants))
            rows = (len(plants) + cols - 1) // cols
            for c in range(cols):
                grid.columnconfigure(c, weight=1, uniform="p")
            for r_ in range(rows):
                grid.rowconfigure(r_, weight=1, uniform="p")
            for i, e in enumerate(plants):
                current = e.get("name") == self.plant_label
                tk.Button(grid, text=e.get("name", "?"), bg=GREEN_DK if current else GREEN, fg="white",
                          activebackground=GREEN_DK, activeforeground="white", bd=0,
                          font=("Helvetica", PX(22), "bold"), wraplength=max(120, self.sw // cols - f(60)),
                          command=lambda e=e: self.pick_plant(e)
                          ).grid(row=i // cols, column=i % cols, sticky="nsew", padx=f(6), pady=f(6))
        tk.Button(body, text="Cancel", bg=GRAY, fg="white", activebackground=GRAY, activeforeground="white",
                  bd=0, font=("Helvetica", PX(14), "bold"), pady=f(8),
                  command=self.close_overlay).pack(fill="x", pady=(f(6), 0))

    # ---------- overlays (used instead of pop-up windows so they work in kiosk mode) ----------
    def open_overlay(self, title):
        self.close_overlay()
        ov = tk.Frame(self.root, bg=CARD)
        ov.place(x=0, y=0, relwidth=1, relheight=1)
        ov.lift()
        tk.Label(ov, text=title, bg=GREEN, fg="white", font=("Helvetica", PX(16), "bold"),
                 pady=self.fs(8)).pack(fill="x")
        body = tk.Frame(ov, bg=CARD)
        body.pack(fill="both", expand=True, padx=self.fs(14), pady=self.fs(10))
        self.overlay = ov
        return body

    def close_overlay(self):
        self.pw_active = False
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

    def popup(self, title, text, on_ok=None):
        body = self.open_overlay(title)
        tk.Label(body, text=text, bg=CARD, fg=TXT, wraplength=self.sw - self.fs(80), justify="center",
                 font=("Helvetica", PX(15))).pack(expand=True)

        def ok():
            self.close_overlay()
            if on_ok:
                on_ok()
        tk.Button(body, text="OK", bg=GREEN, fg="white", activebackground=GREEN, activeforeground="white",
                  bd=0, command=ok, font=("Helvetica", PX(16), "bold"), padx=self.fs(40),
                  pady=self.fs(8)).pack(pady=self.fs(10))

    # ---------- secret exit ----------
    def secret_tap(self):
        now = time.time()
        if now - self.last_tap > TAP_WINDOW_S:
            self.tap_count = 0
        self.last_tap = now
        self.tap_count += 1
        if self.tap_count >= SECRET_TAPS:
            self.tap_count = 0
            if now < self.locked_until:
                self.popup("Locked", f"Too many wrong attempts. Try again in {int(self.locked_until - now)} seconds.")
            else:
                self.ask_password()

    def ask_password(self):
        f = self.fs
        body = self.open_overlay("Enter password")
        var = tk.StringVar()
        tk.Entry(body, textvariable=var, show="*", state="readonly", readonlybackground="#f1f1f1",
                 justify="center", font=("Helvetica", PX(18))).pack(fill="x", pady=(0, f(6)))
        shift = [False]
        letter_btns = []
        kb = tk.Frame(body, bg=CARD)
        kb.pack(expand=True, fill="both")

        def press(ch):
            var.set(var.get() + (ch.upper() if shift[0] else ch))

        rows = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm", "-_@#!.$%&*"]
        for chars in rows:
            fr = tk.Frame(kb, bg=CARD)
            fr.pack(expand=True, fill="both")
            fr.rowconfigure(0, weight=1)
            for ci, ch in enumerate(chars):
                fr.columnconfigure(ci, weight=1, uniform="k")
                b = tk.Button(fr, text=ch, font=("Helvetica", PX(14), "bold"), bg="#e0e0e0", bd=0,
                              command=lambda c=ch: press(c))
                b.grid(row=0, column=ci, sticky="nsew", padx=1, pady=1)
                if ch.isalpha():
                    letter_btns.append((b, ch))

        def toggle_shift():
            shift[0] = not shift[0]
            for b, ch in letter_btns:
                b.config(text=ch.upper() if shift[0] else ch)

        def submit():
            now = time.time()
            if check_password(var.get(), self.cfg):
                self.exit_app()
                return
            self.fails += 1
            if self.fails >= 3:
                self.fails, self.locked_until = 0, now + 30
                self.popup("Locked", "3 wrong attempts. Locked for 30 seconds.")
            else:
                self.popup("Wrong password", "That password is not correct.", on_ok=self.ask_password)

        self.pw_active, self.pw_var, self.pw_submit = True, var, submit
        ctl = tk.Frame(body, bg=CARD)
        ctl.pack(fill="x", pady=(f(6), 0))
        for i, (t, c, cmd) in enumerate([("Shift", GRAY, toggle_shift),
                                         ("Delete", GRAY, lambda: var.set(var.get()[:-1])),
                                         ("Cancel", RED, self.close_overlay), ("OK", GREEN, submit)]):
            ctl.columnconfigure(i, weight=1, uniform="x")
            tk.Button(ctl, text=t, bg=c, fg="white", activebackground=c, activeforeground="white", bd=0,
                      command=cmd, font=("Helvetica", PX(13), "bold"), pady=f(6)
                      ).grid(row=0, column=i, sticky="ew", padx=3)

    def exit_app(self):
        self.cam.halt.set()
        self.root.destroy()


# ----------------------------------------------------------------------------
# install / uninstall / main
# ----------------------------------------------------------------------------
def do_install():
    MODELS_DIR.mkdir(exist_ok=True)
    CAPTURES_DIR.mkdir(exist_ok=True)
    ensure_suggestions_file()
    if "password" not in load_config():
        prompt_new_password()
    AUTOSTART_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTOSTART_PATH.write_text(
        "[Desktop Entry]\nType=Application\nName=AgriLens\n"
        f"Path={APP_DIR}\n"
        f"Exec='{sys.executable}' '{Path(__file__).resolve()}'\n"
        "X-GNOME-Autostart-enabled=true\n")
    print(f"\nStart-at-boot entry written: {AUTOSTART_PATH}")
    print(f"Put each plant's .tflite + labels file in: {MODELS_DIR}")
    print(f"Admin keyboard combo to stop the app: {KEYBOARD_EXIT_COMBO}  (Ctrl+Alt+Shift+Q), then the password")
    print("Next: sudo raspi-config -> System Options -> Boot / Auto Login -> Desktop Autologin, then reboot.")


def main():
    ap = argparse.ArgumentParser(description="AgriLens plant diagnosis kiosk")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--set-password", action="store_true")
    ap.add_argument("--windowed", action="store_true", help="normal window instead of full screen")
    ap.add_argument("--hide-cursor", action="store_true", help="hide the pointer (touchscreen without mouse)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if args.set_password:
        prompt_new_password()
        return
    if args.install:
        do_install()
        return
    if args.uninstall:
        AUTOSTART_PATH.unlink(missing_ok=True)
        print("Start-at-boot entry removed.")
        return

    cfg = load_config()
    if "password" not in cfg:
        print(f"WARNING: no exit password set, default is '{DEFAULT_PASSWORD}'. "
              f"Run: python3 {Path(__file__).name} --set-password")
    if "quality" not in cfg or "min_confidence" not in cfg:
        cfg.setdefault("min_confidence", 0.5)
        cfg.setdefault("quality", QUALITY_DEFAULTS)
        save_config(cfg)

    root = make_root(args)   # window + "starting" text appear immediately...
    load_heavy()             # ...then the slow libraries load
    AgriLensApp(root, cfg, args)
    root.mainloop()


if __name__ == "__main__":
    main()
