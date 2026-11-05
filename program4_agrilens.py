#!/usr/bin/env python3
"""
Program 4: AgriLens - plant leaf diagnosis kiosk for Raspberry Pi 4/5 (NOT for Colab).

  * Full-screen tkinter touch UI: live webcam preview, Capture / Diagnose / Clear buttons
  * OpenCV takes the photo, the .tflite from Program 3 diagnoses it (ai-edge-litert)
  * Farmer can pick any .tflite model (app 'models' folder, USB sticks, home folder)
  * Treatment advice comes from disease_suggestions.json (created on first run, editable)
  * Starts at boot. Can only be closed with the secret button in the top-right corner:
    tap it 7 times quickly, then enter the password.

ONE-TIME SETUP ON THE PI (run once from a terminal, or over VNC/SSH):
  1. Put this file in a folder, e.g. /home/pi/agrilens/
  2. Put your model files in /home/pi/agrilens/models/ :
         tomato_mobilenetv4.tflite  +  tomato_labels.txt   (both from Program 3)
  3. python3 program4_agrilens.py --install
         -> installs dependencies (if pip is blocked it builds its own .venv automatically),
            asks you to set the exit password,
            creates the suggestions JSON and the start-at-boot entry
  4. sudo raspi-config -> System Options -> Boot / Auto Login -> Desktop Autologin
     (also Display Options -> Screen Blanking -> Off)
  5. sudo reboot

OTHER COMMANDS:
  python3 program4_agrilens.py                 run now (full screen)
  python3 program4_agrilens.py --windowed      run in a normal window (testing over VNC)
  python3 program4_agrilens.py --set-password  change the exit password
  python3 program4_agrilens.py --uninstall     remove start-at-boot
  Options: --camera 1 (camera index)   --hide-cursor (touchscreens)
"""
import argparse
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
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CAPTURES_DIR = APP_DIR / "captures"
CONFIG_PATH = APP_DIR / "config.json"
SUGGESTIONS_PATH = APP_DIR / "disease_suggestions.json"
LOGO_PATH = APP_DIR / "logo.png"  # optional: shown in the header if present
AUTOSTART_PATH = Path.home() / ".config" / "autostart" / "agrilens.desktop"
DEFAULT_PASSWORD = "agrilens"  # only used if --install / --set-password was never run

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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from PIL import ImageTk  # noqa: E402
except ImportError:
    sys.exit("Run: sudo apt install python3-pil.imagetk")

# ----------------------------------------------------------------------------
# Disease suggestions (written to disease_suggestions.json on first run; edit freely)
# Keys: plant -> class name (same names as in the <plant>_labels.txt file).
# ----------------------------------------------------------------------------
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


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def ensure_suggestions_file():
    if not SUGGESTIONS_PATH.exists():
        SUGGESTIONS_PATH.write_text(json.dumps(DEFAULT_SUGGESTIONS, indent=2, ensure_ascii=False))
        print(f"Created {SUGGESTIONS_PATH}")


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
# Model
# ----------------------------------------------------------------------------
def find_labels(model_path: Path):
    stem, d = model_path.stem, model_path.parent
    plant = stem.split("_")[0]
    for c in (d / f"{stem.replace('_mobilenetv4', '')}_labels.txt", d / f"{stem}_labels.txt",
              d / f"{stem}.txt", d / f"{plant}_labels.txt", d / "labels.txt"):
        if c.exists():
            return c
    return None


def plant_name(model_path: Path, labels_path: Path):
    s = labels_path.stem
    return s[: -len("_labels")] if s.endswith("_labels") else model_path.stem.split("_")[0]


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
# UI
# ----------------------------------------------------------------------------
GREEN, GREEN_DK = "#2e7d32", "#1b5e20"
BLUE, GRAY = "#1565c0", "#607d8b"
BG, CARD, TXT = "#eef3ee", "#ffffff", "#1b1b1b"
RED, ORANGE = "#c62828", "#e65100"
SECRET_TAPS, TAP_WINDOW_S = 7, 2.5


class AgriLensApp:
    def __init__(self, root, cfg, args):
        self.root, self.cfg, self.args = root, cfg, args
        self.fullscreen = not args.windowed
        self.sw, self.sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.k = max(0.7, self.sh / 480)
        self.db = load_suggestions()
        self.clf, self.plant, self.model_name = None, "", ""
        self.cap, self.live, self.frozen, self.tkimg = None, None, None, None
        self.cam_fail, self.overlay = 0, None
        self.tap_count, self.last_tap, self.fails, self.locked_until = 0, 0.0, 0, 0.0

        root.title("AgriLens")
        root.configure(bg=BG)
        if self.fullscreen:
            root.attributes("-fullscreen", True)
            try:
                root.attributes("-topmost", True)
            except tk.TclError:
                pass
        else:
            root.geometry("800x480")
        root.protocol("WM_DELETE_WINDOW", lambda: None)  # the window can't be closed normally
        if args.hide_cursor:
            root.config(cursor="none")

        self.build_ui()
        self.open_camera()
        last = args.model or cfg.get("last_model")
        if last and Path(last).exists():
            self.load_model(Path(last), quiet=True)
        self.update_status()
        self.tick()

    def fs(self, n):
        return max(8, int(n * self.k))

    # ---------- layout ----------
    def build_ui(self):
        r, fs = self.root, self.fs
        header = tk.Frame(r, bg=GREEN, height=fs(46))
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="AGRILENS  -  Plant Leaf Diagnosis", bg=GREEN, fg="white",
                 font=("Helvetica", fs(15), "bold")).pack(side="left", padx=fs(12))
        # secret button: small, unlabeled, top-right corner
        tk.Button(header, text="\u00b7", bg=GREEN_DK, fg=GREEN_DK, activebackground=GREEN,
                  activeforeground=GREEN, bd=0, highlightthickness=0, relief="flat",
                  width=4, command=self.secret_tap).pack(side="right", fill="y")
        if LOGO_PATH.exists():
            try:
                logo = Image.open(LOGO_PATH)
                logo.thumbnail((fs(40), fs(40)))
                self.logo_img = ImageTk.PhotoImage(logo)
                tk.Label(header, image=self.logo_img, bg=GREEN).pack(side="right", padx=fs(6))
            except Exception:  # noqa: BLE001
                pass
        tk.Button(header, text="Choose Model", bg="#43a047", fg="white", activebackground="#66bb6a",
                  font=("Helvetica", fs(11), "bold"), bd=0, padx=fs(10),
                  command=self.choose_model).pack(side="right", padx=fs(8), pady=fs(6))

        self.status = tk.Label(r, text="", bg="#dfe8df", fg="#333", anchor="w",
                               font=("Helvetica", fs(9)), padx=fs(8))
        self.status.pack(side="bottom", fill="x")

        bar = tk.Frame(r, bg=BG)
        bar.pack(side="bottom", fill="x", padx=fs(10), pady=fs(8))
        for i, (txt, col, cmd) in enumerate([("Capture", BLUE, self.capture),
                                             ("Diagnose", GREEN, self.diagnose),
                                             ("Clear", GRAY, self.clear)]):
            bar.columnconfigure(i, weight=1, uniform="b")
            tk.Button(bar, text=txt, bg=col, fg="white", activebackground=col, bd=0,
                      font=("Helvetica", fs(16), "bold"), command=cmd, pady=fs(10)
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
                                  font=("Helvetica", fs(12)))
        self.photo_lbl.place(relx=0.5, rely=0.5, anchor="center")

        info = tk.Frame(body, bg=CARD, padx=fs(10), pady=fs(6))
        info.grid(row=0, column=1, sticky="nsew")
        wrap = max(200, self.sw // 2 - fs(50))
        tk.Label(info, text="Diagnosis (Probable):", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", fs(10))).pack(fill="x")
        self.diag_lbl = tk.Label(info, text="-", bg=CARD, fg=TXT, anchor="w", justify="left",
                                 wraplength=wrap, font=("Helvetica", fs(20), "bold"))
        self.diag_lbl.pack(fill="x")
        self.conf_lbl = tk.Label(info, text="Confidence: -", bg=CARD, fg=TXT, anchor="w",
                                 font=("Helvetica", fs(12), "bold"))
        self.conf_lbl.pack(fill="x", pady=(fs(2), fs(4)))
        tk.Label(info, text="Possible Cause:", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", fs(10), "bold")).pack(fill="x")
        self.cause_lbl = tk.Label(info, text="-", bg=CARD, fg=TXT, anchor="w", justify="left",
                                  wraplength=wrap, font=("Helvetica", fs(11)))
        self.cause_lbl.pack(fill="x", pady=(0, fs(4)))
        tk.Label(info, text="Suggestion:", bg=CARD, fg="#555", anchor="w",
                 font=("Helvetica", fs(10), "bold")).pack(fill="x")
        tf = tk.Frame(info, bg=CARD)
        tf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(tf, width=fs(26))
        sb.pack(side="right", fill="y")
        self.sugg = tk.Text(tf, bg=CARD, fg=TXT, bd=0, wrap="word", font=("Helvetica", fs(11)),
                            yscrollcommand=sb.set, state="disabled", cursor="arrow")
        self.sugg.pack(side="left", fill="both", expand=True)
        sb.config(command=self.sugg.yview)
        self.reset_result("Point the camera at ONE leaf so it fills the picture, tap Capture, then tap Diagnose.")

    def update_status(self):
        cam = "Camera OK" if self.cap is not None else "Camera NOT found"
        model = f"Model: {self.model_name} ({self.plant})" if self.clf else "Model: none selected"
        self.status.config(text=f"{model}   |   {cam}   |   {self.db['_meta'].get('disclaimer', '')}")

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

    # ---------- camera ----------
    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        idx = self.args.camera
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap, self.cam_fail = cap, 0
        else:
            self.cap = None
            self.photo_lbl.config(image="", text="Camera not found.\nCheck the webcam cable, then tap Capture.")
        self.update_status()

    def show(self, bgr):
        w, h = self.photo_frame.winfo_width(), self.photo_frame.winfo_height()
        if w < 20 or h < 20:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        sc = min(w / iw, h / ih)
        img = Image.fromarray(rgb).resize((max(1, int(iw * sc)), max(1, int(ih * sc))))
        self.tkimg = ImageTk.PhotoImage(img)
        self.photo_lbl.config(image=self.tkimg, text="")

    def tick(self):
        if self.frozen is None and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                self.live, self.cam_fail = frame, 0
                self.show(frame)
            else:
                self.cam_fail += 1
                if self.cam_fail > 30:
                    self.open_camera()
        self.root.after(66, self.tick)

    # ---------- buttons ----------
    def capture(self):
        if self.cap is None:
            self.open_camera()
        if self.cap is None:
            self.popup("Camera problem", "No webcam found. Check the cable and try again.")
            return
        for _ in range(2):  # drop stale buffered frames
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            self.popup("Camera problem", "Could not take a photo. Try again.")
            return
        self.frozen = frame
        self.show(frame)
        self.reset_result("Photo taken. Tap Diagnose.")
        try:
            CAPTURES_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(CAPTURES_DIR / time.strftime("leaf_%Y%m%d_%H%M%S.jpg")), frame)
        except Exception:  # noqa: BLE001
            pass

    def clear(self):
        self.frozen = None
        self.reset_result("Point the camera at ONE leaf so it fills the picture, tap Capture, then tap Diagnose.")

    def diagnose(self):
        if self.frozen is None:
            self.popup("No photo yet", "Tap Capture to take a photo of the leaf first.")
            return
        if self.clf is None:
            self.choose_model("Please choose a model first.")
            return
        self.diag_lbl.config(text="Analysing...", fg=TXT)
        self.root.update_idletasks()
        try:
            preds = self.clf.predict(self.frozen)
        except Exception as e:  # noqa: BLE001
            self.popup("Error", f"Could not analyse the photo:\n{e}")
            return
        label, prob = preds[0]
        entry = find_advice(self.db, self.plant, label)
        name = entry.get("display_name") or label.replace("_", " ").title()
        if prob < self.cfg.get("min_confidence", 0.5):
            self.diag_lbl.config(text="Not sure", fg=ORANGE)
            self.conf_lbl.config(text=f"Confidence: {prob * 100:.0f}% (too low)")
            self.cause_lbl.config(text="The photo may be unclear, or this leaf is not covered by the selected model.")
            self.set_suggestions([
                "Retake the photo: one leaf, filling the picture, in good daylight, plain background.",
                "Make sure the right model is selected for this plant.",
                f"Closest guess: {name} ({prob * 100:.0f}%). Do not rely on it.",
            ])
            return
        healthy = key(label) == "healthy"
        self.diag_lbl.config(text=name, fg=GREEN if healthy else RED)
        self.conf_lbl.config(text=f"Confidence: {prob * 100:.0f}%")
        self.cause_lbl.config(text=entry.get("cause", "-"))
        self.set_suggestions(entry.get("suggestions", []))

    # ---------- model ----------
    def find_models(self):
        found, seen = [], set()
        for base in (MODELS_DIR, Path("/media"), Path("/mnt"), Path.home()):
            if not base.exists():
                continue
            depth0 = len(base.parts)
            for dirpath, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                if len(Path(dirpath).parts) - depth0 >= 3:
                    dirs[:] = []
                for f in files:
                    if f.lower().endswith(".tflite"):
                        p = Path(dirpath) / f
                        if p.resolve() not in seen:
                            seen.add(p.resolve())
                            found.append(p)
        return found

    def load_model(self, path: Path, quiet=False):
        labels = find_labels(path)
        if labels is None:
            self.popup("Labels file missing",
                       f"Put the labels file next to the model, named e.g.\n{path.stem.split('_')[0]}_labels.txt\n"
                       f"(in {path.parent})")
            return False
        try:
            self.clf = Classifier(path, labels)
        except Exception as e:  # noqa: BLE001
            self.clf = None
            self.popup("Could not load model", str(e))
            return False
        self.plant, self.model_name = plant_name(path, labels), path.name
        self.cfg["last_model"] = str(path)
        save_config(self.cfg)
        self.update_status()
        if not quiet:
            self.popup("Model ready", f"{path.name}\nPlant: {self.plant}\nClasses: {len(self.clf.labels)}")
        return True

    def choose_model(self, note=None):
        f = self.fs
        body = self.open_overlay("Choose a model (.tflite)")
        if note:
            tk.Label(body, text=note, bg=CARD, fg=ORANGE, font=("Helvetica", f(12), "bold")).pack(pady=(0, f(4)))
        paths = self.find_models()
        lf = tk.Frame(body, bg=CARD)
        lf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(lf, width=f(30))
        sb.pack(side="right", fill="y")
        lb = tk.Listbox(lf, font=("Helvetica", f(13)), activestyle="none", selectbackground=GREEN,
                        exportselection=False, yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.config(command=lb.yview)
        for p in paths:
            lb.insert("end", f"{p.name}      [{p.parent}]")
        if not paths:
            lb.insert("end", f"No .tflite files found. Copy a model + labels file into {MODELS_DIR}")

        def use():
            sel = lb.curselection()
            if sel and paths:
                self.close_overlay()
                self.load_model(paths[sel[0]])

        def browse():
            from tkinter import filedialog
            if self.fullscreen:
                self.root.attributes("-fullscreen", False)
            p = filedialog.askopenfilename(parent=self.root, title="Choose a .tflite model",
                                           initialdir=str(MODELS_DIR if MODELS_DIR.exists() else Path.home()),
                                           filetypes=[("TFLite model", "*.tflite")])
            if self.fullscreen:
                self.root.attributes("-fullscreen", True)
            if p:
                self.close_overlay()
                self.load_model(Path(p))

        bf = tk.Frame(body, bg=CARD)
        bf.pack(fill="x", pady=(f(8), 0))
        for i, (t, c, cmd) in enumerate([("Use Selected", GREEN, use), ("Browse...", BLUE, browse),
                                         ("Cancel", GRAY, self.close_overlay)]):
            bf.columnconfigure(i, weight=1, uniform="m")
            tk.Button(bf, text=t, bg=c, fg="white", activebackground=c, bd=0, command=cmd,
                      font=("Helvetica", f(14), "bold"), pady=f(8)).grid(row=0, column=i, sticky="ew", padx=f(4))

    # ---------- overlays (used instead of pop-up windows so they work in kiosk mode) ----------
    def open_overlay(self, title):
        self.close_overlay()
        ov = tk.Frame(self.root, bg=CARD)
        ov.place(x=0, y=0, relwidth=1, relheight=1)
        ov.lift()
        tk.Label(ov, text=title, bg=GREEN, fg="white", font=("Helvetica", self.fs(16), "bold"),
                 pady=self.fs(8)).pack(fill="x")
        body = tk.Frame(ov, bg=CARD)
        body.pack(fill="both", expand=True, padx=self.fs(14), pady=self.fs(10))
        self.overlay = ov
        return body

    def close_overlay(self):
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None

    def popup(self, title, text, on_ok=None):
        body = self.open_overlay(title)
        tk.Label(body, text=text, bg=CARD, fg=TXT, wraplength=self.sw - self.fs(80), justify="center",
                 font=("Helvetica", self.fs(15))).pack(expand=True)

        def ok():
            self.close_overlay()
            if on_ok:
                on_ok()
        tk.Button(body, text="OK", bg=GREEN, fg="white", activebackground=GREEN, bd=0, command=ok,
                  font=("Helvetica", self.fs(16), "bold"), padx=self.fs(40), pady=self.fs(8)).pack(pady=self.fs(10))

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
                 justify="center", font=("Helvetica", f(18))).pack(fill="x", pady=(0, f(6)))
        shift = [False]
        letter_btns = []
        kb = tk.Frame(body, bg=CARD)
        kb.pack(expand=True, fill="both")

        def press(ch):
            var.set(var.get() + (ch.upper() if shift[0] else ch))

        rows = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm", "-_@#!.$%&*"]
        for r, chars in enumerate(rows):
            fr = tk.Frame(kb, bg=CARD)
            fr.pack(expand=True, fill="both")
            for ci, ch in enumerate(chars):
                fr.columnconfigure(ci, weight=1, uniform="k")
                b = tk.Button(fr, text=ch, font=("Helvetica", f(14), "bold"), bg="#e0e0e0", bd=0,
                              command=lambda c=ch: press(c))
                b.grid(row=0, column=ci, sticky="nsew", padx=1, pady=1)
                if ch.isalpha():
                    letter_btns.append((b, ch))
            fr.rowconfigure(0, weight=1)

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

        ctl = tk.Frame(body, bg=CARD)
        ctl.pack(fill="x", pady=(f(6), 0))
        for i, (t, c, cmd) in enumerate([("Shift", GRAY, toggle_shift),
                                         ("Delete", GRAY, lambda: var.set(var.get()[:-1])),
                                         ("Cancel", RED, self.close_overlay), ("OK", GREEN, submit)]):
            ctl.columnconfigure(i, weight=1, uniform="x")
            tk.Button(ctl, text=t, bg=c, fg="white", activebackground=c, bd=0, command=cmd,
                      font=("Helvetica", f(13), "bold"), pady=f(6)).grid(row=0, column=i, sticky="ew", padx=3)

    def exit_app(self):
        if self.cap is not None:
            self.cap.release()
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
        f"Exec=sh -c \"sleep 5; exec '{sys.executable}' '{Path(__file__).resolve()}' --hide-cursor\"\n"
        "X-GNOME-Autostart-enabled=true\n")
    print(f"\nStart-at-boot entry written: {AUTOSTART_PATH}")
    print(f"Put your model(s) + labels in: {MODELS_DIR}")
    print("Next: sudo raspi-config -> System Options -> Boot / Auto Login -> Desktop Autologin, then reboot.")


def main():
    ap = argparse.ArgumentParser(description="AgriLens plant diagnosis kiosk")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--set-password", action="store_true")
    ap.add_argument("--windowed", action="store_true", help="normal window instead of full screen")
    ap.add_argument("--hide-cursor", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", type=Path, default=None)
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
    root = tk.Tk()
    AgriLensApp(root, cfg, args)
    root.mainloop()


if __name__ == "__main__":
    main()
