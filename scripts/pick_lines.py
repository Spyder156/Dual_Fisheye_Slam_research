#!/usr/bin/env python3
"""
Straight-line picker (arm C full-FoV physical constraint).

Real straight edges (table edges, door frames, ceiling grid, floor-tile seams)
MUST unproject to a single plane through the camera center (a great circle).
Clicking points along real edges — from near center out to the rim — constrains
the lens model at EVERY theta, not just the seam. Feeds the joint fitter together
with the overlap correspondences.

Shows cam1 | cam2 side by side, with theta=90 (green) / 100 (red) guide circles
and a zoom loupe. Each line belongs to ONE lens (all its points on one side).

CONTROLS
  left-click : add a point to the CURRENT line (stay on one side per line)
  n          : finish current line, start a new one   (needs >= 3 points)
  u          : undo last point (or delete last finished line if current is empty)
  c          : save all lines and exit
  q / ESC    : quit WITHOUT saving

Run:
  python scripts/pick_lines.py --fid 729
Saves -> outputs/armC/manual_lines/lines_<fid>.json   (resumes if it exists)
"""
import os, json, argparse, numpy as np, cv2

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
OUTD = f"{ROOT}/outputs/armC/manual_lines"; os.makedirs(OUTD, exist_ok=True)
CAM1 = dict(fx=998.71442825746783, fy=1006.4807034467711, cx=1612.0, cy=1617.0,
            k=[0.034358190499964164, -0.01780110435914365, -0.00079837513258086007, 0.00036177684527421565])
CAM2 = dict(fx=997.51132933461327, fy=1005.2394553958525, cx=1612.0, cy=1617.0,
            k=[0.036087615485948625, -0.02348048096992612, 0.0042046925204551316, -0.00090948827268945594])
DISP_W = 900; LOUPE = 150
COLORS = [(0,0,255),(0,255,0),(255,128,0),(0,255,255),(255,0,255),(255,255,0),
          (128,0,255),(0,128,255),(0,255,128),(200,200,200),(80,127,255),(255,80,180)]


def theta_radius(cam, theta):
    k1,k2,k3,k4 = cam["k"]; t2=theta*theta
    return cam["fx"]*theta*(1+k1*t2+k2*t2**2+k3*t2**3+k4*t2**4)

def draw_guides(img, cam, sc):
    c=(int(cam["cx"]*sc), int(cam["cy"]*sc))
    cv2.drawMarker(img, c, (255,255,255), cv2.MARKER_CROSS, 14, 1)
    for th,col in [(np.pi/2,(0,220,0)), (np.deg2rad(100),(0,0,220))]:
        cv2.circle(img, c, int(theta_radius(cam,th)*sc), col, 2)
    return img


class LinePicker:
    def __init__(self, fid):
        self.fid=fid
        f1=[f for f in os.listdir(f"{FIR}/cam1") if f"_{fid}_" in f][0]
        f2=[f for f in os.listdir(f"{FIR}/cam2") if f"_{fid}_" in f][0]
        self.full1=cv2.imread(f"{FIR}/cam1/{f1}"); self.full2=cv2.imread(f"{FIR}/cam2/{f2}")
        self.H,self.W=self.full1.shape[:2]; self.sc=DISP_W/self.W
        dh=int(self.H*self.sc)
        self.base1=draw_guides(cv2.resize(self.full1,(DISP_W,dh)).copy(),CAM1,self.sc)
        self.base2=draw_guides(cv2.resize(self.full2,(DISP_W,dh)).copy(),CAM2,self.sc)
        self.dh=dh
        self.lines=[]            # [{cam:1/2, pts:[(u,v)...]}]
        self.cur=[]; self.cur_cam=None
        self.cursor=(0,0)
        ex=f"{OUTD}/lines_{fid}.json"
        if os.path.exists(ex):
            ed=json.load(open(ex))
            for L in ed.get("lines",[]):
                self.lines.append({"cam":L["cam"],"pts":[tuple(p) for p in L["pts"]]})
            print(f"[resume] loaded {len(self.lines)} existing lines")

    def d2f(self,x,y):
        if x<DISP_W: return 1, x/self.sc, y/self.sc
        return 2, (x-DISP_W)/self.sc, y/self.sc
    def f2d(self,cam,u,v):
        return int(u*self.sc + (0 if cam==1 else DISP_W)), int(v*self.sc)

    def on_mouse(self,ev,x,y,flags,_):
        self.cursor=(x,y)
        if ev==cv2.EVENT_LBUTTONDOWN:
            cam,u,v=self.d2f(x,y)
            if not (0<=u<self.W and 0<=v<self.H): return
            if self.cur_cam is None: self.cur_cam=cam
            if cam!=self.cur_cam: return          # keep a line on one lens
            self.cur.append((u,v))

    def loupe(self,canvas):
        x,y=self.cursor
        if x<DISP_W: full,ox,uf,vf=self.full1,0,x/self.sc,y/self.sc
        else: full,ox,uf,vf=self.full2,DISP_W,(x-DISP_W)/self.sc,y/self.sc
        uf,vf=int(uf),int(vf); x0,y0=max(0,uf-LOUPE//2),max(0,vf-LOUPE//2)
        crop=full[y0:y0+LOUPE,x0:x0+LOUPE]
        if crop.size==0: return
        z=cv2.resize(crop,(LOUPE*2,LOUPE*2),interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(z,(LOUPE,LOUPE),(0,255,255),cv2.MARKER_CROSS,22,1)
        cv2.rectangle(z,(0,0),(LOUPE*2-1,LOUPE*2-1),(255,255,255),2)
        canvas[0:LOUPE*2, ox:ox+LOUPE*2]=z

    def render(self):
        c=np.hstack([self.base1.copy(),self.base2.copy()])
        for i,L in enumerate(self.lines):
            col=COLORS[i%len(COLORS)]
            pts=[self.f2d(L["cam"],u,v) for u,v in L["pts"]]
            for j in range(1,len(pts)): cv2.line(c,pts[j-1],pts[j],col,1)
            for p in pts: cv2.drawMarker(c,p,col,cv2.MARKER_CROSS,10,1)
            if pts: cv2.putText(c,str(i),(pts[0][0]+6,pts[0][1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
        if self.cur:
            pts=[self.f2d(self.cur_cam,u,v) for u,v in self.cur]
            for j in range(1,len(pts)): cv2.line(c,pts[j-1],pts[j],(0,255,255),1)
            for p in pts: cv2.drawMarker(c,p,(0,255,255),cv2.MARKER_TILTED_CROSS,12,1)
        self.loupe(c)
        b=f"frame {self.fid} | lines:{len(self.lines)} | current pts:{len(self.cur)} (cam{self.cur_cam}) | click=add  n=finish line  u=undo  c=save  q=quit"
        cv2.rectangle(c,(0,self.dh-28),(c.shape[1],self.dh),(0,0,0),-1)
        cv2.putText(c,b,(8,self.dh-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        return c

    def finish_line(self):
        if len(self.cur)>=3:
            self.lines.append({"cam":self.cur_cam,"pts":self.cur})
        self.cur=[]; self.cur_cam=None

    def undo(self):
        if self.cur: self.cur.pop();  self.cur_cam=self.cur_cam if self.cur else None
        elif self.lines: self.lines.pop()

    def save(self):
        if len(self.cur)>=3: self.finish_line()
        out=dict(fid=self.fid,cam1=CAM1,cam2=CAM2,
                 lines=[{"cam":L["cam"],"pts":[[u,v] for u,v in L["pts"]]} for L in self.lines])
        p=f"{OUTD}/lines_{self.fid}.json"; json.dump(out,open(p,'w'),indent=2)
        npts=sum(len(L['pts']) for L in self.lines)
        print(f"saved {len(self.lines)} lines ({npts} points) -> {p}")

    def run(self):
        win="line picker  (cam1 | cam2)"; cv2.namedWindow(win,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win,self.on_mouse)
        while True:
            cv2.imshow(win,self.render()); k=cv2.waitKey(20)&0xFF
            if k==ord('n'): self.finish_line()
            elif k==ord('u'): self.undo()
            elif k==ord('c'): self.save(); break
            elif k in (ord('q'),27): print("quit without saving"); break
        cv2.destroyAllWindows()


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--fid",default="729"); a=ap.parse_args()
    LinePicker(a.fid).run()
