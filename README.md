# JHAROKHA-AI — Heritage Surface Screening Prototype

**Question:** Can a computer-vision workflow make candidate lines visible on heritage surfaces while being honest about the risk of confusing damage with decoration?

JHAROKHA-AI is a Flask + OpenCV student research prototype. It accepts a surface photograph, runs a simple edge-detection pipeline, overlays candidate edges, and records a local field log.

## What it does now
- Segments cracks with a trained U-Net (if `weights/crack_unet.pt` exists) or a ridge-filter baseline (if not).
- Measures total length, max/mean width, cracked area %, branch points and crack direction.
- Optional: enter the real width of the photographed area (cm) to get sizes in mm.
- Gives severity (hairline / moderate / severe), a 0-100 condition score, a risk tier and an indicative re-inspection interval.
- Charts: condition gauge, width histogram, direction chart, and a growth projection built from repeat photos of the same site label.

## Important limitations
- The condition score, tier and interval are rule-based screening heuristics (thresholds are at the top of `analysis.py`). They are not engineering values.
- A single photo cannot tell you how many years a structure will last. The projection chart uses *your repeat photos over time* (same site label, same framing) and shows a trend, not remaining life.
- Carvings, jali, joints and shadows can still be mistaken for cracks, so review the overlay. Do not use for structural or conservation decisions.

## One-click path (Windows)
1. Put crack-free photos in `my_photos\no_cracks` (carved stone, arches, jali, plain walls).
2. Double-click `train_now.bat`. It sets up, downloads public data, and trains.
3. Double-click `run_app.bat` and open http://127.0.0.1:5000.

## Train your own model (VS Code terminal)
```powershell
pip install -r requirements-train.txt
python prepare_data.py deepcrack          # downloads public DeepCrack (537 images) as pretraining data
python prepare_data.py folder --images D:\photos --masks D:\masks --site haveliA   # your labelled photos
python prelabel.py --images D:\photos --out prelabel\haveliA   # draft masks to fix in Paint/GIMP
python prepare_data.py folder --images D:\jali --site jaliB --negatives   # crack-free photos, no masks needed
python prepare_data.py stats              # shows whether your validation split is trustworthy
python train.py --epochs 1 --limit 16     # 1-2 min smoke test, run this first
python train.py                           # full run
```
Slow CPU? `python train.py --encoder resnet18 --crop 256 --crops-per-image 2 --epochs 15`.

Rules that matter more than any setting:
- One site name per building/wall (`haveliA`, `jaliB`...). Validation holds out whole sites, so label at least 3-4 different ones.
- Label your own heritage photos, including crack-free ornament (jali, carvings, joints) as hard negatives. Public datasets are concrete and asphalt, so they teach cracks but not decoration.
- Every epoch prints tolerance-F1, IoU and the false-alarm rate on crack-free images. Training saves the best weights plus the best threshold to `weights/crack_unet.pt`, and a full log to `weights/train_log.json` for your paper. Restart `app.py` to use them.

## Run on Windows (PowerShell)
1. Install Python 3.11 or 3.12 from the official Python website and enable the option to add Python to PATH.
2. Open this folder in VS Code.
3. In VS Code, choose **Terminal → New Terminal**. Make sure the terminal is in the folder containing `app.py`.
4. Run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.
6. To stop the local server, click the terminal and press `Ctrl+C`.

If PowerShell blocks activation, run this command in the same terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This changes the policy only for that PowerShell process.

## Research path: damage vs decoration
1. Build a labelled image dataset with two kinds of examples: suspected genuine cracks and hard negatives such as carved grooves, jali openings, mortar joints, scratches, stains, shadows, and weathered texture.
2. Have more than one reviewer label the images and document disagreements. Avoid claiming expert ground truth without appropriate review.
3. Split train/test data by building or site, not by randomly splitting near-identical photos, to reduce data leakage.
4. Compare this baseline with a suitable pretrained segmentation/detection model.
5. Report precision, recall, F1, false-positive rate on ornamental surfaces, and results broken down by surface type. Include uncertainty or human-review routing for ambiguous cases.
6. Keep structural safety assessment outside the app's claims.

## Project structure

```text
jharokha-ai/
├── app.py
├── analysis.py                  # segmentation, measurements, severity, projection
├── train.py                     # train the U-Net locally
├── crack_data.py                # crops, augmentation, metrics
├── prepare_data.py              # build the dataset (DeepCrack + your photos)
├── requirements-train.txt
├── data/images, data/masks      # your training data
├── weights/crack_unet.pt        # created by train.py
├── requirements.txt
├── README.md
├── .gitignore
├── history.json                 # created after first successful analysis; ignored by Git
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── about.html
└── static/
    ├── style.css
    ├── uploads/                 # uploaded images; ignored by Git
    └── results/                 # overlays; ignored by Git
```

## GitHub: publish under `ashglitch`
Create an empty repository named `jharokha-ai` under your GitHub account. Do not add a README, license, or `.gitignore` on GitHub because this project already has those files. Then open a terminal in this project folder and run:

```powershell
git init
git add .
git commit -m "Initial JHAROKHA-AI heritage screening prototype"
git branch -M main
git remote add origin https://github.com/ashglitch/jharokha-ai.git
git push -u origin main
```

If Git asks you to configure an identity, use your own GitHub name and email:

```powershell
git config --global user.name "Your GitHub Name"
git config --global user.email "your-github-email@example.com"
```

Never put passwords, access tokens, or private keys in this repository. GitHub may ask you to sign in through a browser.
