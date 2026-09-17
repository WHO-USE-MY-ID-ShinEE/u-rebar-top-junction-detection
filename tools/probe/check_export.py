# -*- coding: utf-8 -*-
import sys, io, json
sys.path.insert(0, r"D:\work\gygj")
from src import pipeline
pipeline.run_detect(1)
print("--- station_1_points.csv ---")
print(io.open(r"D:\work\gygj\outputs\station_1_points.csv", encoding="utf-8").read()[:400])
print("--- station_1_points.json ---")
d = json.load(io.open(r"D:\work\gygj\outputs\station_1_points.json", encoding="utf-8"))
print("station", d["station"], "点数", len(d["points"]), "前5个", d["points"][:5])
