"""Study 2, exploratory: how well could a *true* map answer "inside" if it were blurred by the same amount of error
as the decoded map? Adds isotropic Gaussian noise to the true positions so that R² matches a given value, then
answers rung 4 from the noisy positions. No model or GPU needed."""
import numpy as np, pandas as pd
from geomap.triangles import generate_tri_scene, in_triangle
from geomap.study2 import balanced_accuracy
rng = np.random.default_rng(0)
pool = [f"n{i}" for i in range(40)]
tss = [generate_tri_scene(rng, pool) for _ in range(3000)]
pts = np.array([[ts.scene.coords[n] for n in (*ts.corners, ts.point)] for ts in tss], float)   # (n,4,2)
cen = pts - pts.mean(1, keepdims=True)
var = cen.reshape(-1, 2).var(0).mean()
print(f"scene-centred coord sd {np.sqrt(var):.2f} units")
hard = np.array([ts.stratum in ("inside", "box_hard") for ts in tss])
y = np.array([ts.stratum == "inside" for ts in tss])[hard].astype(int)
for r2 in (0.62, 0.74, 0.85, 0.9, 0.95, 0.98):
    sd = np.sqrt((1 - r2) * var)
    noisy = pts + rng.normal(0, sd, pts.shape)
    pred = np.array([in_triangle(tuple(p[3]), tuple(p[0]), tuple(p[1]), tuple(p[2])) for p in noisy])[hard].astype(int)
    print(f"R² {r2:.2f}: noise sd {sd:.2f} units -> inside BA (hard) {balanced_accuracy(y, pred, 1):.2f}, says yes {pred.mean():.2f}")
