# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
for n in [1, 5, 10]:
    tif = imread_any(os.path.join(data, f"station_{n}_output_depth_raw.tif")).astype(np.float32)
    d = cv2.rotate(tif, cv2.ROTATE_90_COUNTERCLOCKWISE)
    fin = np.isfinite(d) & (d > 0)
    near = fin & (d < 0.83)
    far = fin & (d >= 0.83)
    def runs(mask, axis):
        m = mask if axis == 0 else mask.T
        out = []
        for c in range(m.shape[1]):
            col = m[:, c].astype(np.int8)
            dif = np.diff(np.concatenate([[0], col, [0]]))
            st = np.where(dif == 1)[0]; en = np.where(dif == -1)[0]
            out.extend((en - st).tolist())
        return np.array(out)
    rn = runs(near, 0); rf = runs(far, 1)
    print(f"st{n}: near-horizontal-bar vertical thickness px  median={np.median(rn):.1f} "
          f"p25={np.percentile(rn,25):.0f} p75={np.percentile(rn,75):.0f} n={rn.size}")
    print(f"       far-vertical-bar  horizontal thickness px  median={np.median(rf):.1f} "
          f"p25={np.percentile(rf,25):.0f} p75={np.percentile(rf,75):.0f} n={rf.size}")
    print(f"       near depth median={np.median(d[near]):.3f} m, far={np.median(d[far]):.3f} m, "
          f"gap={(np.median(d[far])-np.median(d[near]))*1000:.1f} mm")
