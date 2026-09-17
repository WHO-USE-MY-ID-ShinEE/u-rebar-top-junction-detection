# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\work\gygj")
from src import pipeline
for n in (1, 7):
    r = pipeline.run_detect(n)
    print(n, r["stats"], "->", r["out"])
