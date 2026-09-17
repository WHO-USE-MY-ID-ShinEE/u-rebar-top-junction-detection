import importlib, sys
for m in ["cv2", "numpy", "scipy", "skimage", "matplotlib", "PIL", "pandas", "sklearn"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} OK  {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({type(e).__name__})")
