#!/usr/bin/env python3
"""
Visualize the overlap ray-disagreement so it can be inspected by eye.
For each manual correspondence: where cam1's ray lands (red +) and where cam2's
ray lands (cyan x) on the equirect, plus an EXAGGERATED arrow (residual x SCALE)
so the pattern is visible. Structured/global pattern => convention/extrinsic;
scattered/epipolar => parallax. Base FIORD model + COLMAP extrinsic.
Output -> outputs/armC/investigate/overlap_residuals.png
"""
import os, re, json, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/home/raghav/workspace/INSV_STITCHING"
FIR=f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
BTXT=f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
PAIRS=f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
OUT=f"{ROOT}/outputs/armC/investigate"; os.makedirs(OUT,exist_ok=True)
FID="729"; EQW=4096; EQH=2048; FOV=200.0; SCALE=15   # arrow exaggeration
ROLL=np.array([[np.cos(np.pi),-np.sin(np.pi),0],[np.sin(np.pi),np.cos(np.pi),0],[0,0,1]])

def qR(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def R12c():
    R1,R2={},{}
    for l in open(BTXT):
        if l.startswith('#') or not l.strip():continue
        p=l.split()
        if len(p)<10 or not p[0].isdigit():continue
        m=re.search(r'_(\d+)_fisheye(\d)',p[9])
        if m:(R1 if m.group(2)=='1' else R2)[m.group(1)]=qR(*map(float,p[1:5]))
    rels=np.array([R2[k]@R1[k].T for k in set(R1)&set(R2)]);U,_,Vt=np.linalg.svd(rels.mean(0));R=U@Vt
    if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
    return R
def unproj(u,v,c):
    xp=(u-c['cx'])/c['fx'];yp=(v-c['cy'])/c['fy'];thd=np.hypot(xp,yp);th=thd.copy()
    for _ in range(18):
        poly=1+sum(c['k'][j]*th**(2*(j+1)) for j in range(len(c['k'])));dp=sum(c['k'][j]*2*(j+1)*th**(2*(j+1)-1) for j in range(len(c['k'])))
        th=th-(th*poly-thd)/(poly+th*dp)
    phi=np.arctan2(yp,xp);return np.stack([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)],-1)
def proj(X,Y,Z,c):
    r=np.sqrt(X*X+Y*Y);th=np.arctan2(r,Z);poly=1+sum(c['k'][j]*th**(2*(j+1)) for j in range(len(c['k'])))
    s=np.where(r>1e-9,th*poly/np.where(r>1e-9,r,1),0.);return c['fx']*X*s+c['cx'],c['fy']*Y*s+c['cy'],th
def dirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi;lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi;lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
def dpx(d,W,H):
    X,Y,Z=d[...,0],d[...,1],d[...,2];lon=np.arctan2(X,Z);lat=np.arcsin(np.clip(-Y,-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H
def render(f1,f2,c1,c2,R12,W,H):
    D=dirs(W,H)
    def pr(f,c,R):
        d=D@R.T;u,v,th=proj(d[...,0],d[...,1],d[...,2],c);val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        return cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0)),val,th
    e1,v1,t1=pr(f1,c1,ROLL);e2,v2,t2=pr(f2,c2,R12@ROLL);u1=v1&(~v2|(t1<=t2));u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1);comb[u1]=e1[u1];comb[u2]=e2[u2];return comb

d=json.load(open(PAIRS));c1,c2=d['cam1'],d['cam2']
P=np.array([[p['u1'],p['v1'],p['u2'],p['v2']] for p in d['pairs']])
R12=R12c()
f1=cv2.imread(f"{FIR}/cam1/"+[f for f in os.listdir(f'{FIR}/cam1') if f'_{FID}_' in f][0])
f2=cv2.imread(f"{FIR}/cam2/"+[f for f in os.listdir(f'{FIR}/cam2') if f'_{FID}_' in f][0])
eq=render(f1,f2,c1,c2,R12,EQW,EQH)
r1=unproj(P[:,0],P[:,1],c1);r2=unproj(P[:,2],P[:,3],c2)
d1=r1@ROLL.T; d2=(r2@R12)@ROLL.T
x1,y1=dpx(d1,EQW,EQH);x2,y2=dpx(d2,EQW,EQH)
res_deg=np.degrees(np.arccos(np.clip((r1*(r2@R12)).sum(1),-1,1)))

fig,ax=plt.subplots(figsize=(24,12));ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
for i in range(len(P)):
    # exaggerated arrow from cam1-landing toward cam2-landing
    dx,dy=(x2[i]-x1[i])*SCALE,(y2[i]-y1[i])*SCALE
    ax.arrow(x1[i],y1[i],dx,dy,head_width=18,head_length=22,fc='yellow',ec='yellow',lw=1.2,length_includes_head=True,zorder=4)
    ax.plot(x1[i],y1[i],'+',c='red',ms=9,mew=1.1,zorder=5)
    ax.text(x1[i]+10,y1[i]-10,f"{i}:{res_deg[i]:.1f}",color='white',fontsize=8,zorder=6)
ax.set_title(f"Overlap ray disagreement (arrows x{SCALE})  base model  median {np.median(res_deg):.2f} deg  "
             f"[red+ = cam1 ray, arrow -> cam2 ray]",fontsize=13);ax.axis('off')
fig.tight_layout();fig.savefig(f"{OUT}/overlap_residuals.png",dpi=130,bbox_inches='tight');plt.close(fig)
print(f"WROTE {OUT}/overlap_residuals.png  (median {np.median(res_deg):.2f} deg, arrows x{SCALE})")
print("per-pair deg:",np.round(res_deg,2).tolist())
