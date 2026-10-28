r"""See what the trained model "thinks" of one photo, at several zoom levels.

    python diagnose.py path\to\photo.jpg

Prints, for each zoom, how confident the model is, and saves heat-maps to diagnose_out\
(bright = "this looks like a crack"). Useful when a photo with visible cracks gets "No cracks found".
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import analysis

if len(sys.argv) < 2:
    raise SystemExit('Usage: python diagnose.py path\\to\\photo.jpg')
bgr = cv2.imread(sys.argv[1])
if bgr is None:
    raise SystemExit('Could not read that image.')
h, w = bgr.shape[:2]
if max(h, w) > analysis.WORK_SIZE:
    s = analysis.WORK_SIZE / max(h, w)
    bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
if analysis._load_model() is None:
    raise SystemExit('No trained model loaded (weights/crack_unet.pt missing or torch not installed).')
out = Path('diagnose_out'); out.mkdir(exist_ok=True)
print(f"threshold used by the app: {analysis._model['thr']}")
print('zoom  max-prob  %pixels>0.3  >0.5  >0.7')
for s, p in analysis.model_probs(bgr).items():
    print(f'{s:>4}x  {p.max():8.3f}  {100*(p>.3).mean():10.2f}  {100*(p>.5).mean():5.2f}  {100*(p>.7).mean():5.2f}')
    heat = cv2.applyColorMap((cv2.resize(p, (bgr.shape[1], bgr.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / f'heat_{s}x.jpg'), cv2.addWeighted(bgr, .4, heat, .6, 0))
print('heat-maps saved in diagnose_out\\')
