"""Pre-label your photos so labelling takes minutes, not hours.

    python prelabel.py --images D:\\photos\\haveli --out prelabel\\haveliA

Creates:
    prelabel/haveliA/masks/<name>.png    draft masks (white = crack) from the current detector
    prelabel/haveliA/preview/<name>.jpg  overlay so you can see what it found

Then FIX the masks in Paint / GIMP / Photopea (paint black over false alarms, white over missed cracks).
Crack-free photo? Make the mask fully black (or skip the mask and use --negatives when importing).
Import the corrected set:
    python prepare_data.py folder --images D:\\photos\\haveli --masks prelabel\\haveliA\\masks --site haveliA
Retrain, run prelabel again with the better model, fix less each round.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import analysis
import crack_data

ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
out = Path(a.out)
(out / 'masks').mkdir(parents=True, exist_ok=True)
(out / 'preview').mkdir(parents=True, exist_ok=True)
n = 0
for ip in sorted(Path(a.images).iterdir()):
    if ip.suffix.lower() not in crack_data.EXTS:
        continue
    bgr = cv2.imread(str(ip))
    if bgr is None:
        continue
    h, w = bgr.shape[:2]
    if max(h, w) > analysis.WORK_SIZE:
        s = analysis.WORK_SIZE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    mask, method, _ = analysis.segment(bgr)
    cv2.imwrite(str(out / 'masks' / f'{ip.stem}.png'), mask.astype(np.uint8) * 255)
    prev = bgr.copy(); prev[mask] = (35, 65, 220)
    cv2.imwrite(str(out / 'preview' / f'{ip.stem}.jpg'), cv2.addWeighted(bgr, .55, prev, .45, 0))
    n += 1
print(f'{n} draft masks written to {out} using: {method}')
print('Masks are at 1024 px scale; the importer resizes them to match your photos automatically.')
