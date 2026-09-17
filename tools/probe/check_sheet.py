# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, r"D:\work\gygj")
from src import imgio
d = imgio.imread_any(r"D:\work\gygj\outputs\station_6_result.png")
print("station_6_result.png 尺寸:", d.shape, "类型:", d.dtype)
print("颜色统计: 红点像素(近似) =", int(((d[:,:,2]>200)&(d[:,:,1]<60)&(d[:,:,0]<60)).sum()))
print("白底占比:", round(float((d>240).all(axis=2).mean()), 3))
for n in (1, 6, 10):
    p = rf"D:\work\gygj\outputs\station_{n}_result.png"
    print(f"station_{n}_result.png  {os.path.getsize(p)/1e6:.2f} MB")
