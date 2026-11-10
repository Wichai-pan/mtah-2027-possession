"""Render a clip of the tactical video with the labels and the tracking drawn on it.

    uv run --group viz python scripts/render_overlay.py                       # public 5 minutes
    uv run --group viz python scripts/render_overlay.py --start 40000 --end 42500 \
        --pred results/submission_v2_firstpass.csv --out results/viz/clip.mp4

On every frame:
  - the tracking, projected into the image with the frame's homography: home slots blue, away
    slots red, the ball a yellow ring ("BALL NOT TRACKED" when missing)
  - top bar: the ground truth (GT) and our prediction (PRED) at this frame, and the
    Smart Tagging instances active at this time
  - bottom: a +/-15 s timeline, one row for GT and one for PRED, with the current frame marked;
    a GT possession change flashes for one second

Output only stays local (results/ is gitignored): the video is not redistributable.
"""

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from mtah.official import possession_metrics as pm
from mtah.schema import Match

DATA = Path("../repo/data/tactical")
SRC_W, SRC_H, FPS = 1920, 1080, 25
COL = {0: (200, 110, 20), 1: (40, 40, 220), 2: (150, 150, 150), 3: (40, 40, 40)}   # BGR: H A D X
NAME = {0: "BARCELONA (H)", 1: "ATHLETIC (A)", 2: "DEAD BALL (D)", 3: "not graded (X)"}


def load_homography(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, 10))
    return arr.reshape(-1, 3, 3)


def project(Hm: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = np.c_[xy, np.ones(len(xy))] @ Hm.T
    return pts[:, :2] / pts[:, 2:3]


def smart_tags(path: Path):
    out = []
    for ins in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in ins.findall("label")}
        out.append((round(float(ins.findtext("start")) * FPS), round(float(ins.findtext("end")) * FPS),
                    ins.findtext("code"), tags.get("Team", ""), tags.get("Type", "")))
    return out


def read_labels(path):
    return None if path is None else pm.label_codes(pm.read_label_csv(str(path)))


LUT = np.array([COL[k] for k in range(4)], np.uint8)


def timeline(w, t, rows, span, mark=None):
    """rows: [(name, codes)] -> image strip, one row per label track, centred on frame t."""
    h_row = 34
    img = np.full((h_row * len(rows) + 22, w, 3), 25, np.uint8)
    x0 = 70
    frames = np.arange(t - span, t + span + 1)
    for r, (name, codes) in enumerate(rows):
        y = r * h_row + 4
        cv2.putText(img, name, (6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        if codes is None:
            continue
        valid = (frames >= 0) & (frames < len(codes))
        c = np.full(len(frames), 3)
        c[valid] = codes[frames[valid]]
        strip = LUT[c][None]                                   # (1, n, 3)
        img[y:y + h_row - 8, x0:w - 10] = cv2.resize(strip, (w - 10 - x0, h_row - 8),
                                                     interpolation=cv2.INTER_NEAREST)
    for s in range(-span // FPS, span // FPS + 1, 5):          # tick every 5 s
        x = x0 + int((s * FPS + span) * (w - x0 - 10) / len(frames))
        cv2.line(img, (x, img.shape[0] - 20), (x, img.shape[0] - 14), (200, 200, 200), 1)
        cv2.putText(img, f"{s:+d}s", (x - 12, img.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)
    xc = x0 + int(span * (w - x0 - 10) / len(frames))
    cv2.line(img, (xc, 0), (xc, img.shape[0] - 20), (255, 255, 255), 2)
    if mark is not None and abs(mark - t) <= span:
        xm = x0 + int((mark - t + span) * (w - x0 - 10) / len(frames))
        cv2.line(img, (xm, 0), (xm, img.shape[0] - 20), (80, 220, 80), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=11185 - 50)
    ap.add_argument("--end", type=int, default=18684 + 50)
    ap.add_argument("--gt", default="../repo/evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--pred", default="results/submission_v2_firstpass.csv")
    ap.add_argument("--npz", default="../external_data/mtah_bar_ath.npz")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--span-s", type=int, default=15)
    ap.add_argument("--out", default="results/viz/public_5min_overlay.mp4")
    ap.add_argument("--mark", type=int, default=None,
                    help="a reference frame (e.g. a tag start): shown as a green line and a t-mark clock")
    args = ap.parse_args()

    m = Match.load(args.npz)
    Hs = load_homography(DATA / "BAR-ATH_tactical_homography.csv")
    tags = smart_tags(DATA / "BAR-ATH_tactical_smart_tagging.xml")
    gt, pred = read_labels(args.gt), read_labels(args.pred if args.pred else None)
    W = args.width
    Hh = round(SRC_H * W / SRC_W / 2) * 2
    sc = W / SRC_W
    span = args.span_s * FPS
    rows = [("GT", gt), ("PRED", pred)]
    panel_h = timeline(W, args.start, rows, span).shape[0]
    top_h = 92
    out_h = top_h + Hh + panel_h
    out_h += out_h % 2

    n = args.end - args.start + 1
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{args.start / FPS:.3f}", "-i", str(DATA / "BAR-ATH_tactical.mp4"),
         "-frames:v", str(n), "-vf", f"scale={W}:{Hh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{out_h}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)

    last_change = -10 ** 9
    change_txt = ""
    for i in range(n):
        raw = dec.stdout.read(W * Hh * 3)
        if len(raw) < W * Hh * 3:
            break
        t = args.start + i
        frame = np.frombuffer(raw, np.uint8).reshape(Hh, W, 3).copy()

        # tracking projected into the image
        if Hs.shape[0] > t and np.isfinite(Hs[t]).all():
            xy = m.player_xy[t]
            ok = ~np.isnan(xy[:, 0])
            if ok.any():
                for (px, py), team in zip(project(Hs[t], xy[ok]) * sc, m.slot_team[ok]):
                    cv2.circle(frame, (int(px), int(py)), 7, COL[int(team)], 2, cv2.LINE_AA)
            if not np.isnan(m.ball_xy[t, 0]):
                bx, by = project(Hs[t], m.ball_xy[t:t + 1])[0] * sc
                cv2.circle(frame, (int(bx), int(by)), 11, (0, 230, 255), 2, cv2.LINE_AA)

        canvas = np.full((out_h, W, 3), 15, np.uint8)
        canvas[top_h:top_h + Hh] = frame
        g = int(gt[t]) if gt is not None else 3
        p = int(pred[t]) if pred is not None else 3
        cv2.rectangle(canvas, (0, 0), (W // 2 - 2, 40), COL[g], -1)
        cv2.rectangle(canvas, (W // 2 + 2, 0), (W, 40), COL[p], -1)
        cv2.putText(canvas, f"GT: {NAME[g]}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PRED: {NAME[p]}", (W // 2 + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        mm, ss = divmod(t / FPS, 60)
        active = [f"{c} [{tm}{' ' + ty if ty else ''}]" for a, b, c, tm, ty in tags if a <= t <= b]
        info = f"frame {t}   video {int(mm):02d}:{ss:05.2f}"
        if args.mark is not None:
            info += f"   t - tag start = {(t - args.mark) / FPS:+.1f}s"
        cv2.putText(canvas, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        tag_txt = "smart tagging: " + (" | ".join(active) if active else "-")
        cv2.putText(canvas, tag_txt[:170], (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
        if np.isnan(m.ball_xy[t, 0]):
            cv2.putText(canvas, "BALL NOT TRACKED", (W - 220, top_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 230, 255), 2, cv2.LINE_AA)

        if gt is not None and t > 0 and gt[t] != gt[t - 1] and gt[t] != 3 and gt[t - 1] != 3:
            last_change, change_txt = t, f"GT CHANGE: {'HAD'[gt[t - 1]]} -> {'HAD'[gt[t]]}"
        if t - last_change < FPS:
            cv2.putText(canvas, change_txt, (20, top_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(canvas, change_txt, (20, top_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL[int(gt[t])], 2, cv2.LINE_AA)

        canvas[top_h + Hh:top_h + Hh + panel_h] = timeline(W, t, rows, span, args.mark)
        enc.stdin.write(canvas.tobytes())
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)

    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
