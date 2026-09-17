"""
Rebar Intersection Detection v8 - skeleton + crossing + midpoint depth
========================================================================
Builds on v7 (depth-ridge centre refinement) and changes the crossing depth
to the MIDPOINT of the two crossing bar depths:

    z_cross = (z_outer + z_inner) / 2

Pipeline (identical to v7 until the depth step):
  1. filter_top_layer(): nearest-to-camera rebar layer (the top grid of a
     multi-layer cage) -> Zhang-Suen skeleton.
  2. HoughLinesP on the skeleton -> cluster -> PCA fit -> crossing points.
  3. Depth-ridge centre refinement re-fits each bar on the round bar top
     surface, so the crossing lands on the bar's geometric centre.
  4. Depth: z_outer (top-bar surface) and z_inner (bottom-bar surface) are
     resolved from the depth map in a window; the reported crossing depth is
     their midpoint (z_cross), i.e. between the horizontal and vertical bar.
"""

import numpy as np
import cv2
from pathlib import Path

from detection_final_v4 import (_to_mm, filter_top_layer, _align_gray_to_depth,
                                _imread_any)
from detection_final_v6 import detect_rebar_intersections
from detection_final_v7 import (_refine_bar, _line_to_bar, _clip_seg,
                                _intersect_lines)


# ---------------------------------------------------------------------------
# Intersection depth: midpoint of the two crossing bar depths
# ---------------------------------------------------------------------------
def intersection_depths_v8(depth_image, intersections, R=15, rebar_d=16.0):
    """Return (z_outer, z_inner, z_cross) [mm] per intersection.

    - z_outer: top-bar surface (nearest-to-camera layer).
    - z_inner: bottom-bar surface (contact plane), falling back to
               z_outer + rebar_d when the bottom bar is fully occluded.
    - z_cross: (z_outer + z_inner) / 2, the midpoint of the two bar depths.
    """
    mm = _to_mm(depth_image)
    top = filter_top_layer(mm)
    h, w = mm.shape

    out = []
    for ix, iy in intersections:
        y0, y1 = max(0, iy - R), min(h, iy + R + 1)
        x0, x1 = max(0, ix - R), min(w, ix + R + 1)
        vals = mm[y0:y1, x0:x1][top[y0:y1, x0:x1]]
        vals = vals[np.isfinite(vals) & (vals > 0)]

        if len(vals) < 20:
            v = mm[iy, ix]
            z_outer = float(v) if (np.isfinite(v) and v > 0) else float("nan")
            z_inner = z_outer + rebar_d
        else:
            vals = np.sort(vals)
            vmin = float(vals.min())
            near = vals[vals < vmin + 15.0]
            z_outer = float(np.median(near))

            deeper = vals[vals > z_outer + 10.0]
            z_inner = float(np.median(deeper)) if len(deeper) >= 8 else z_outer + rebar_d

        z_cross = (z_outer + z_inner) / 2.0
        out.append((z_outer, z_inner, z_cross))

    return out


def detect_rebar_intersections_v8(depth_image, gray_image=None, **kwargs):
    """v6 detection + depth-ridge refine + midpoint crossing depth.

    Returns (intersections, depths, h_lines, v_lines, skeleton_mask);
    depths[i] == (z_outer, z_inner, z_cross) in mm.
    """
    rebar_d = kwargs.pop("rebar_d", 16.0)
    x_offset = kwargs.pop("x_offset", 0.0)
    y_offset = kwargs.pop("y_offset", 0.0)

    intersections, h_lines, v_lines, skeleton_mask = \
        detect_rebar_intersections(depth_image, gray_image, **kwargs)

    mm = _to_mm(depth_image)
    top = filter_top_layer(mm)
    h, w = mm.shape
    sk = skeleton_mask > 0

    # refine each bar's centre-line from the depth ridge
    refined_h = []
    for l in h_lines:
        r = _refine_bar(mm, top, l, True)
        refined_h.append(r if r is not None else _line_to_bar(l))
    refined_v = []
    for l in v_lines:
        r = _refine_bar(mm, top, l, False)
        refined_v.append(r if r is not None else _line_to_bar(l))

    def is_crossing(ix, iy, R=12, strip=4):
        has_h = sk[max(0, iy - strip):min(h, iy + strip + 1),
                   max(0, ix - R):min(w, ix + R + 1)].any()
        has_v = sk[max(0, iy - R):min(h, iy + R + 1),
                   max(0, ix - strip):min(w, ix + strip + 1)].any()
        return has_h and has_v

    new_pts = []
    for a in refined_h:
        for b in refined_v:
            p = _intersect_lines(a, b)
            ix, iy = int(round(p[0])), int(round(p[1]))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            if not is_crossing(ix, iy):
                continue
            new_pts.append((ix + int(round(x_offset)), iy + int(round(y_offset))))

    h_line_segs = [_clip_seg(b) for b in refined_h]
    v_line_segs = [_clip_seg(b) for b in refined_v]

    depths = intersection_depths_v8(depth_image, new_pts, rebar_d=rebar_d)
    return new_pts, depths, h_line_segs, v_line_segs, skeleton_mask


def run_detection_station_v8(data_dir, stations=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                             out_dir=None, rebar_d=16.0,
                             x_offset=0.0, y_offset=0.0):
    data_path = Path(data_dir)
    out_path = Path(out_dir) if out_dir else data_path

    print("=" * 64)
    print("Rebar Intersection Detection v8 (skeleton + crossing + midpoint depth)")
    print("=" * 64)

    summary = []
    for n in stations:
        tif_path = data_path / f"station_{n}_output_depth_raw.tif"
        png_path = data_path / f"station_{n}_output_image_left.png"
        if not tif_path.exists():
            print(f"  skip station {n}: depth file missing")
            continue

        print(f"\n--- Station {n} ---")
        depth = _imread_any(tif_path)
        gray = _imread_any(png_path) if png_path.exists() else None
        if depth is None:
            print(f"  skip station {n}: failed to read depth")
            continue

        intersections, depths, h_lines, v_lines, skeleton_mask = \
            detect_rebar_intersections_v8(depth, gray, rebar_d=rebar_d,
                                          x_offset=x_offset, y_offset=y_offset)

        print(f"H-lines: {len(h_lines)}, V-lines: {len(v_lines)}")
        print(f"Intersections: {len(intersections)}")
        summary.append((n, len(h_lines), len(v_lines), len(intersections)))

        csv_path = out_path / f"station_{n}_tie_points_v8.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("point_id,x_px,y_px,z_outer_mm,z_inner_mm,z_cross_mm\n")
            for i, ((ix, iy), (zo, zi, zc)) in enumerate(zip(intersections, depths)):
                f.write(f"P{i + 1:03d},{ix},{iy},{zo:.2f},{zi:.2f},{zc:.2f}\n")
        print(f"Saved: {csv_path}")

        for i, ((ix, iy), (zo, zi, zc)) in enumerate(zip(intersections, depths)):
            print(f"  {i + 1:2d}. ({ix:4d},{iy:4d}) z_outer={zo:.1f} "
                  f"z_inner={zi:.1f} z_cross={zc:.1f}mm")

        if gray is not None:
            disp = _align_gray_to_depth(gray, depth.shape)
        else:
            disp = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        vis = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        mask_color = np.zeros_like(vis)
        mask_color[skeleton_mask > 0] = [0, 40, 0]
        vis = cv2.addWeighted(vis, 0.85, mask_color, 0.15, 0)
        for x1, y1, x2, y2 in h_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 0), 6, cv2.LINE_AA)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (220, 0, 0), 6, cv2.LINE_AA)
        for i, ((ix, iy), (zo, zi, zc)) in enumerate(zip(intersections, depths)):
            cv2.circle(vis, (ix, iy), 12, (0, 0, 255), -1)
            cv2.circle(vis, (ix, iy), 14, (255, 255, 255), 2)
            cv2.putText(vis, f"{i + 1}:{zc:.0f}", (ix + 16, iy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"Intersections: {len(intersections)} "
                         f"(H:{len(h_lines)} V:{len(v_lines)})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out = out_path / f"station_{n}_redetect_v8.png"
        cv2.imencode(".png", vis)[1].tofile(str(out))
        print(f"Saved: {out}")

    print("\nSummary (station, H, V, intersections):")
    for s in summary:
        print(f"  {s[0]}: H={s[1]} V={s[2]} I={s[3]}")
    print("\nDone!")


if __name__ == "__main__":
    run_detection_station_v8(
        r"D:\rebar-tie\RVCProject_full_20260814_1630\RVCProject")
