# -*- coding: utf-8 -*-
import cv2, numpy as np, os
def imread_any(p, f=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)
data = r"D:\work\gygj\任务书数据+代码"
out = r"D:\work\gygj\_probe\preview"
png = imread_any(os.path.join(data, "station_1_output_image_left.png"))
g = png if png.ndim == 2 else cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
print("gray stats: mean=%.1f med=%.0f p1=%.0f p5=%.0f p50=%.0f p95=%.0f p99=%.0f max=%d"
      % (g.mean(), np.median(g), *np.percentile(g,[1,5,50,95,99]), g.max()))
print("frac<20:", round(float((g<20).mean()),3), " frac>200:", round(float((g>200).mean()),3))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(g)
# gamma on normalized
gn = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)/255
gam = (np.power(gn, 0.45)*255).astype(np.uint8)
trio = np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                  cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
                  cv2.cvtColor(gam, cv2.COLOR_GRAY2BGR)])
cv2.putText(trio,"raw",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
cv2.putText(trio,"CLAHE",(W:=g.shape[1]+20,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
cv2.putText(trio,"gamma0.45",(2*g.shape[1]+20,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
cv2.imwrite(os.path.join(out,"gray_enh.png"), cv2.resize(trio,(trio.shape[1]//2, trio.shape[0]//2)))
print("saved")
