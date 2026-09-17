# -*- coding: utf-8 -*-
import zipfile, os
z = zipfile.ZipFile(r"D:\work\gygj\智能系统工程训练任务书20260907.docx")
out = r"D:\work\gygj\_probe\media"
os.makedirs(out, exist_ok=True)
for m in z.namelist():
    if "media/" in m and not m.endswith("/"):
        with open(os.path.join(out, os.path.basename(m)), "wb") as f:
            f.write(z.read(m))
        print("saved", m)
