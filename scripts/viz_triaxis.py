#!/usr/bin/env python3
"""
Draw a 3D coordinate GIZMO in the center of one equirect: world axes
X=red, Y=green, Z=blue as arrows from a central origin, foreshortened by the
camera pose (orthographic). Axis pointing toward viewer = solid; away = dashed+faint.
Lets you read the world orientation directly (e.g. which axis is 'up'/ceiling).
Output -> outputs/armC/investigate/triaxis_<fid>.png
"""
import os, re, json, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/home/raghav/workspace/INSV_STITCHING"
FIR=f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
BTXT=f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
LINES=f"{ROOT}/outputs/armC/manual_lines/lines_729.json"
OUT=f"{ROOT}/outputs/armC/investigate"; os.makedirs(OUT,exist_ok=True)
EQW=2048; EQH=1024; FOV=200.0
ROLL=np.array([[np.cos(np.pi),-np.sin(np.pi),0],[np.sin(np.pi),np.cos(np.pi),0],[0,0,1]])

def qR(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def load_poses():
    R1,R2={},{}
    for l in open(BTXT):
        if l.startswith('#') or not l.strip():continue
        p=l.split()
        if len(p)<10 or not p[0].isdigit():continue
        m=re.search(r'_(\d+)_fisheye(\d)',p[9])
        if m:(R1 if m.group(2)=='1' else R2)[m.group(1)]=qR(*map(float,p[1:5]))
    return R1,R2
def R12_from(R1,R2):
    rels=np.array([R2[k]@R1[k].T for k in set(R1)&set(R2)]);U,_,Vt=np.linalg.svd(rels.mean(0));R=U@Vt
    if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
    return R
def proj(X,Y,Z,c):
    r=np.sqrt(X*X+Y*Y);th=np.arctan2(r,Z);poly=1+sum(c['k'][j]*th**(2*(j+1)) for j in range(len(c['k'])))
    s=np.where(r>1e-9,th*poly/np.where(r>1e-9,r,1),0.);return c['fx']*X*s+c['cx'],c['fy']*Y*s+c['cy'],th
def dirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi;lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi;lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
def render(f1,f2,c1,c2,R12,W,H):
    D=dirs(W,H)
    def pr(f,c,R):
        d=D@R.T;u,v,th=proj(d[...,0],d[...,1],d[...,2],c);val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        return cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0)),val,th
    e1,v1,t1=pr(f1,c1,ROLL);e2,v2,t2=pr(f2,c2,R12@ROLL);u1=v1&(~v2|(t1<=t2));u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1);comb[u1]=e1[u1];comb[u2]=e2[u2];return comb
def img(fid,cam):
    return cv2.imread(f"{FIR}/{cam}/"+[f for f in os.listdir(f'{FIR}/{cam}') if f'_{fid}_' in f][0])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fid",default="729"); a=ap.parse_args()
    R1s,R2s=load_poses(); R12=R12_from(R1s,R2s); ld=json.load(open(LINES)); c1,c2=ld['cam1'],ld['cam2']
    eq=render(img(a.fid,'cam1'),img(a.fid,'cam2'),c1,c2,R12,EQW,EQH)
    R1=R1s[a.fid]
    cx,cy=EQW/2,EQH/2; L=260
    axes=[('X',np.array([1,0,0]),'red'),('Y',np.array([0,1,0]),'lime'),('Z',np.array([0,0,1]),'deepskyblue')]

    fig,ax=plt.subplots(figsize=(20,10)); ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
    ax.scatter([cx],[cy],s=40,c='white',zorder=6)
    report=[]
    for name,w,col in axes:
        d=ROLL@(R1@w)                      # world axis in equirect frame  (x right, y down, z fwd)
        # orthographic gizmo: screen offset = (dx, dy); dz = toward(+)/away(-) viewer
        ex,ey=cx+L*d[0], cy+L*d[1]
        solid = d[2]>=0
        ax.annotate("",xy=(ex,ey),xytext=(cx,cy),zorder=7,
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=3.5,alpha=1 if solid else 0.45,
                                    linestyle='-' if solid else (0,(4,3))))
        ax.text(ex+8*np.sign(d[0]+1e-9),ey+8*np.sign(d[1]+1e-9),
                f"+{name}"+("" if solid else " (away)"),color=col,fontsize=16,weight='bold',zorder=8)
        # human-readable direction
        horiz='right' if d[0]>0.2 else ('left' if d[0]<-0.2 else '')
        vert='down' if d[1]>0.2 else ('up' if d[1]<-0.2 else '')
        depth='toward' if d[2]>0.2 else ('away' if d[2]<-0.2 else '')
        report.append(f"+{name} -> {' '.join(filter(None,[vert,horiz,depth])) or 'center'}")
    ax.set_title(f"World-axis gizmo  frame {a.fid}   (X=red Y=green Z=blue; solid=toward, dashed=away)  "
                 f"top of image = up",fontsize=14); ax.axis('off')
    fig.tight_layout(); fig.savefig(f"{OUT}/triaxis_{a.fid}.png",dpi=120,bbox_inches='tight'); plt.close(fig)
    print("axis directions in image:", " | ".join(report))
    print(f"WROTE {OUT}/triaxis_{a.fid}.png")

if __name__=="__main__":
    main()
