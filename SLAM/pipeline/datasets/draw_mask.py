#!/usr/bin/env python
"""Draw a self-occlusion mask by hand. Once per rig, ~5 minutes.

Uses MATPLOTLIB for the window, not cv2.imshow: the conda OpenCV here is the
headless build (no GTK/Qt), and pip-installing a GUI OpenCV into someone's
environment to draw a polygon is not a reasonable trade.

WHY BY HAND. The occluder is fixed hardware in a fixed place -- it does not move
and does not vary between sequences, so there is nothing to discover. Two
attempts at deriving it from temporal variance both failed, because "static"
equally describes the black region outside the image circle and any distant
wall. The field does not derive it either: #7 used the organisers' mask, #2 drew
their own.

WHAT IT IS FOR. Features on the rig's own hardware are worse than no features:
the hardware is static in the camera frame, so it reports "no motion" while the
rig is moving, and that wrong signal fights the correct one from the other lens.

CONTROLS
  left click   place polygon points
  ENTER        accept this polygon and start another
  s            SAVE this camera and move to the next
  ESC          clear everything
  q            skip this camera without saving

The image is shown UPRIGHT (the Hilti rig is mounted inverted), but the mask is
written in RAW sensor orientation, because that is what the extractor sees.
Getting that backwards would silently mask the wrong half of the image.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

for _b in ("QtAgg", "TkAgg"):
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import PolygonSelector  # noqa: E402

# matplotlib binds 's' to "save figure", 'q' to quit, 'p' to pan -- which is why
# pressing s opened a file dialog instead of saving the mask.
for _k in ("keymap.save", "keymap.quit", "keymap.pan", "keymap.zoom",
           "keymap.home", "keymap.back", "keymap.forward", "keymap.fullscreen",
           "keymap.grid", "keymap.yscale", "keymap.xscale"):
    try:
        matplotlib.rcParams[_k] = []
    except KeyError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from display import ROT180, show  # noqa: E402


def pick_polygons(backgrounds, title):
    """Click a polygon; ENTER accepts it and starts a fresh one; 's' saves.

    A PolygonSelector cannot be reset by touching its internals -- doing so
    leaves the old vertices live, so every ENTER re-adds the same polygon and
    the overlay just gets redder. The only reliable reset is to disconnect the
    widget and build a new one.
    """
    polys = []
    fig, ax = plt.subplots(figsize=(11, 11))
    bg = {"i": 0}
    im = ax.imshow(backgrounds[0][1], cmap="gray")
    ax.set_axis_off()
    def retitle():
        ax.set_title(f"{title}   [{backgrounds[bg['i']][0]}]\n"
                     "click points  |  ENTER = accept polygon  |  N = next image  "
                     "|  S = save & next cam  |  ESC = clear  |  Q = skip")
    retitle()
    holder = {"sel": None, "saved": False}

    def new_selector():
        if holder["sel"] is not None:
            holder["sel"].disconnect_events()
            holder["sel"].set_visible(False)
        holder["sel"] = PolygonSelector(ax, lambda v: None, useblit=False)

    def commit():
        sel = holder["sel"]
        v = list(sel.verts) if sel is not None else []
        if len(v) >= 3:
            polys.append(v)
            ax.add_patch(plt.Polygon(v, closed=True, fill=True,
                                     facecolor="red", alpha=0.35,
                                     edgecolor="yellow", lw=2))
            return True
        return False

    def onkey(ev):
        if ev.key == "enter":
            if commit():
                new_selector()
                fig.canvas.draw_idle()
        elif ev.key in ("s", "S"):
            commit()                     # include whatever is on screen
            holder["saved"] = True
            plt.close(fig)
        elif ev.key == "escape":
            polys.clear()
            for pt_ in list(ax.patches):
                pt_.remove()
            new_selector()
            fig.canvas.draw_idle()
        elif ev.key in ("n", "N", "b", "B"):
            # step through several views of the SAME lens. The occluder is in
            # all of them; the scene is not -- which is what makes it findable.
            bg["i"] = (bg["i"] + (1 if ev.key in ("n", "N") else -1)) % len(backgrounds)
            im.set_data(backgrounds[bg["i"]][1])
            retitle()
            fig.canvas.draw_idle()
        elif ev.key in ("q", "Q"):
            polys.clear()
            plt.close(fig)

    new_selector()
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.tight_layout()
    plt.show()
    return polys if holder["saved"] else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=None,
                    help="frame id to draw on; default picks one mid-sequence")
    ap.add_argument("--scale", type=float, default=0.55)
    a = ap.parse_args()

    ds = Path(a.dataset); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr = np.genfromtxt(ds / "frames.csv", delimiter=",", names=True)
    fid = np.atleast_1d(fr["frame"]).astype(int)
    fsel = a.frame if a.frame is not None else int(fid[len(fid) // 2])

    for c in (0, 1):
        raw = cv2.imread(str(ds / f"cam{c}" / f"{fsel:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"cam{c}: frame {fsel} missing"); continue
        H, W = raw.shape
        disp_full = show(raw)

        # Candidate canvases. The MEDIAN over the sequence is the best one to
        # draw on: the occluder is in every frame so it stays sharp, while the
        # moving scene averages into mush. Brightest frames beat dark ones --
        # frame 4500 is in a blackout, which is the worst possible canvas.
        samp = fid[np.linspace(0, len(fid) - 1, 60).astype(int)]
        stack, means = [], []
        for i in samp:
            im_ = cv2.imread(str(ds / f"cam{c}" / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im_ is not None:
                stack.append(im_); means.append((im_.mean(), i))
        med = np.median(np.stack(stack), axis=0).astype(np.uint8) if stack else raw
        means.sort(reverse=True)
        bgs = [("MEDIAN of sequence (occluder sharp, scene blurred)", show(med))]
        for mval, i in means[:3]:
            im_ = cv2.imread(str(ds / f"cam{c}" / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            bgs.append((f"frame {i} (brightest, mean {mval:.0f})", show(im_)))
        bgs.append((f"frame {fsel} (requested)", disp_full))
        bgs = [(n, cv2.resize(v, None, fx=a.scale, fy=a.scale)) for n, v in bgs]

        polys = pick_polygons(bgs, f"cam{c}")
        if not polys:
            print(f"cam{c}: nothing drawn, skipped"); continue

        # display coords -> full-res display -> RAW sensor orientation
        m_disp = np.zeros(disp_full.shape[:2], np.uint8)
        for p in polys:
            q = (np.array(p, np.float32) / a.scale).astype(np.int32)
            cv2.fillPoly(m_disp, [q], 255)
        m_raw = cv2.rotate(m_disp, cv2.ROTATE_180) if ROT180 else m_disp
        cv2.imwrite(str(out / f"selfocc_cam{c}.png"), m_raw)

        chk = cv2.cvtColor(show(raw), cv2.COLOR_GRAY2BGR)
        chk[show(m_raw) > 0] = 0
        cv2.imwrite(str(out / f"selfocc_cam{c}_applied.png"),
                    cv2.resize(chk, None, fx=0.5, fy=0.5))
        print(f"cam{c}: masked {100*(m_raw>0).mean():.1f}%  -> {out}/selfocc_cam{c}.png")


if __name__ == "__main__":
    main()
