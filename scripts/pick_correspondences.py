#!/usr/bin/env python3
"""
Manual overlap-band correspondence picker (arm C self-calibration frontend).

Shows cam1 | cam2 side by side. You click the SAME world point in each lens's
peripheral overlap band; the pairs are saved for the backend model fitter.

Guide circles mark theta=90 deg (green) and 100 deg (red) — the real overlap
band lives between them. A zoom loupe follows the cursor for precise clicks.

CONTROLS
  left-click   : place/refine the current point (must be on the expected side)
  y            : confirm current point (left first, then its right match)
  u            : undo last confirmed pair (or cancel a half-made one)
  c            : save all pairs and exit
  q / ESC      : quit WITHOUT saving

Run (needs a display):
  python scripts/pick_correspondences.py --fid 729
Saves -> outputs/armC/manual_pairs/pairs_<fid>.json
"""
import os, json, argparse, numpy as np, cv2

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
OUTD = f"{ROOT}/outputs/armC/manual_pairs"; os.makedirs(OUTD, exist_ok=True)

CAM1 = dict(fx=998.71442825746783, fy=1006.4807034467711, cx=1612.0, cy=1617.0,
            k=[0.034358190499964164, -0.01780110435914365, -0.00079837513258086007, 0.00036177684527421565])
CAM2 = dict(fx=997.51132933461327, fy=1005.2394553958525, cx=1612.0, cy=1617.0,
            k=[0.036087615485948625, -0.02348048096992612, 0.0042046925204551316, -0.00090948827268945594])
DISP_W = 900          # display width per fisheye
LOUPE = 160           # loupe source size (px in full-res), shown 2x
COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0),
          (0,128,255),(128,0,255),(0,255,128),(255,128,0)]


def theta_radius(cam, theta):
    """pixel radius from center for a given theta (KB)."""
    k1,k2,k3,k4 = cam["k"]; t2=theta*theta
    td = theta*(1+k1*t2+k2*t2**2+k3*t2**3+k4*t2**4)
    return cam["fx"]*td  # approx (fx~fy)


def draw_guides(img, cam):
    c = (int(cam["cx"]), int(cam["cy"]))
    for th, col in [(np.pi/2,(0,220,0)), (np.deg2rad(100),(0,0,220))]:
        cv2.circle(img, c, int(theta_radius(cam, th)), col, 3)
    return img


class Picker:
    def __init__(self, fid):
        self.fid = fid
        f1 = [f for f in os.listdir(f"{FIR}/cam1") if f"_{fid}_" in f][0]
        f2 = [f for f in os.listdir(f"{FIR}/cam2") if f"_{fid}_" in f][0]
        self.full1 = cv2.imread(f"{FIR}/cam1/{f1}")
        self.full2 = cv2.imread(f"{FIR}/cam2/{f2}")
        self.H, self.W = self.full1.shape[:2]
        self.scale = DISP_W / self.W
        self.base1 = draw_guides(cv2.resize(self.full1, (DISP_W, int(self.H*self.scale))).copy(), self._sc(CAM1))
        self.base2 = draw_guides(cv2.resize(self.full2, (DISP_W, int(self.H*self.scale))).copy(), self._sc(CAM2))
        self.dh = self.base1.shape[0]
        self.pairs = []          # list of ((u1,v1),(u2,v2)) in FULL-res coords
        # RESUME: load existing pairs for this frame so you append, not redo
        existing = f"{OUTD}/pairs_{fid}.json"
        if os.path.exists(existing):
            ed = json.load(open(existing))
            for p in ed.get("pairs", []):
                self.pairs.append(((p["u1"], p["v1"]), (p["u2"], p["v2"])))
            print(f"[resume] loaded {len(self.pairs)} existing pairs from {existing}")
        self.expect = 'L'
        self.left_hold = None    # confirmed left point (full-res) awaiting right
        self.pending = None      # (side, u_full, v_full)
        self.cursor = (0, 0)

    def _sc(self, cam):
        c = dict(cam); c = {**cam, "fx":cam["fx"]*self.scale, "fy":cam["fy"]*self.scale,
                            "cx":cam["cx"]*self.scale, "cy":cam["cy"]*self.scale, "k":cam["k"]}
        return c

    def disp_to_full(self, x, y):
        """canvas (x,y) -> (side, u_full, v_full) or None if in gap."""
        if x < DISP_W:
            return ('L', x/self.scale, y/self.scale)
        elif x >= DISP_W:
            return ('R', (x-DISP_W)/self.scale, y/self.scale)
        return None

    def full_to_disp(self, side, u, v):
        x = u*self.scale + (0 if side=='L' else DISP_W)
        return int(x), int(v*self.scale)

    def on_mouse(self, ev, x, y, flags, _):
        self.cursor = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            r = self.disp_to_full(x, y)
            if r is None: return
            side, u, v = r
            if side != self.expect:
                return  # must click expected side
            self.pending = (side, u, v)

    def loupe(self, canvas):
        x, y = self.cursor
        if x < DISP_W:
            full, ox = self.full1, 0
            uf, vf = x/self.scale, y/self.scale
        else:
            full, ox = self.full2, DISP_W
            uf, vf = (x-DISP_W)/self.scale, y/self.scale
        uf, vf = int(uf), int(vf)
        x0, y0 = max(0, uf-LOUPE//2), max(0, vf-LOUPE//2)
        crop = full[y0:y0+LOUPE, x0:x0+LOUPE]
        if crop.size == 0: return
        z = cv2.resize(crop, (LOUPE*2, LOUPE*2), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(z, (LOUPE, LOUPE), (0,255,255), cv2.MARKER_CROSS, 24, 1)
        cv2.rectangle(z, (0,0), (LOUPE*2-1, LOUPE*2-1), (255,255,255), 2)
        canvas[0:LOUPE*2, ox:ox+LOUPE*2] = z  # top-left of the active half

    def render(self):
        c = np.hstack([self.base1.copy(), self.base2.copy()])
        # saved pairs
        for i,((u1,v1),(u2,v2)) in enumerate(self.pairs):
            col = COLORS[i % len(COLORS)]
            p1 = self.full_to_disp('L',u1,v1); p2 = self.full_to_disp('R',u2,v2)
            for p in (p1,p2):
                cv2.circle(c, p, 6, col, 2); cv2.putText(c, str(i), (p[0]+7,p[1]-7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # left hold (awaiting right)
        if self.left_hold is not None:
            p = self.full_to_disp('L', *self.left_hold)
            cv2.circle(c, p, 7, (0,255,0), 2); cv2.drawMarker(c, p, (0,255,0), cv2.MARKER_TILTED_CROSS, 18, 2)
        # pending
        if self.pending is not None:
            side,u,v = self.pending
            p = self.full_to_disp(side,u,v)
            cv2.drawMarker(c, p, (0,255,255), cv2.MARKER_CROSS, 20, 2)
        self.loupe(c)
        banner = f"frame {self.fid} | pairs:{len(self.pairs)} | EXPECT:{'LEFT' if self.expect=='L' else 'RIGHT match'} | click->y confirm | u undo | c save+quit | q quit"
        cv2.rectangle(c, (0, self.dh-30), (c.shape[1], self.dh), (0,0,0), -1)
        cv2.putText(c, banner, (8, self.dh-9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        return c

    def confirm(self):
        if self.pending is None: return
        side,u,v = self.pending
        if self.expect=='L' and side=='L':
            self.left_hold = (u,v); self.expect='R'; self.pending=None
        elif self.expect=='R' and side=='R':
            self.pairs.append((self.left_hold,(u,v)))
            self.left_hold=None; self.expect='L'; self.pending=None

    def undo(self):
        if self.left_hold is not None:
            self.left_hold=None; self.expect='L'; self.pending=None
        elif self.pairs:
            self.pairs.pop()

    def save(self):
        out = dict(fid=self.fid, cam1=CAM1, cam2=CAM2,
                   pairs=[dict(u1=p[0][0],v1=p[0][1],u2=p[1][0],v2=p[1][1]) for p in self.pairs])
        path = f"{OUTD}/pairs_{self.fid}.json"
        json.dump(out, open(path,'w'), indent=2)
        print(f"saved {len(self.pairs)} pairs -> {path}")

    def run(self):
        win = "overlap picker  (cam1 | cam2)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('y'),): self.confirm()
            elif k in (ord('u'),): self.undo()
            elif k in (ord('c'),): self.save(); break
            elif k in (ord('q'), 27): print("quit without saving"); break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", default="729")
    a = ap.parse_args()
    Picker(a.fid).run()
