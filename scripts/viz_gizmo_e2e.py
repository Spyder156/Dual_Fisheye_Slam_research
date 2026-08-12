#!/usr/bin/env python3
"""
END-TO-END directional test. Draw a real 3D world-axis triad in front of EACH
lens, project it through that lens's KB model + pose onto the FISHEYE image,
THEN run the actual stitch. If the whole pipeline is directionally correct, both
gizmos appear in the equirect agreeing on world directions (Y up in both, X/Z at
the same world walls). Gizmos are drawn on the fisheyes, not the equirect.
Output -> outputs/armC/investigate/gizmo_e2e_<fid>.png
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
def load_pose_t():
    """id -> (R,t) for cam1 & cam2 (world->cam)."""
    P1,P2={},{}
    for l in open(BTXT):
        if l.startswith('#') or not l.strip():continue
        p=l.split()
        if len(p)<10 or not p[0].isdigit():continue
        m=re.search(r'_(\d+)_fisheye(\d)',p[9])
        if m:
            R=qR(*map(float,p[1:5])); t=np.array(list(map(float,p[5:8])))
            (P1 if m.group(2)=='1' else P2)[m.group(1)]=(R,t)
    return P1,P2
def R12_from(P1,P2):
    rels=np.array([P2[k][0]@P1[k][0].T for k in set(P1)&set(P2)]);U,_,Vt=np.linalg.svd(rels.mean(0));R=U@Vt
    if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
    return R
def projf(P,c):   # 3D cam point -> fisheye pixel (KB)
    X,Y,Z=P; r=np.sqrt(X*X+Y*Y); th=np.arctan2(r,Z); poly=1+sum(c['k'][j]*th**(2*(j+1)) for j in range(len(c['k'])))
    s=(th*poly/r) if r>1e-9 else 0.
    return c['fx']*X*s+c['cx'], c['fy']*Y*s+c['cy'], th
def gdirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi;lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi;lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
def dpx1(d,W,H):
    d=d/np.linalg.norm(d);lon=np.arctan2(d[0],d[2]);lat=np.arcsin(np.clip(-d[1],-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H
def draw_curve(ax,dirs,W,H,color,lw=2,alpha=0.9):
    xs=[];ys=[];prev=None
    for d in dirs:
        x,y=dpx1(d,W,H)
        if prev is not None and abs(x-prev)>W/2:
            ax.plot(xs,ys,color=color,lw=lw,alpha=alpha,zorder=6);xs=[];ys=[]
        xs.append(x);ys.append(y);prev=x
    ax.plot(xs,ys,color=color,lw=lw,alpha=alpha,zorder=6)
def proje(X,Y,Z,c):
    r=np.sqrt(X*X+Y*Y);th=np.arctan2(r,Z);poly=1+sum(c['k'][j]*th**(2*(j+1)) for j in range(len(c['k'])))
    s=np.where(r>1e-9,th*poly/np.where(r>1e-9,r,1),0.);return c['fx']*X*s+c['cx'],c['fy']*Y*s+c['cy'],th
def render(f1,f2,c1,c2,R12,W,H):
    D=gdirs(W,H)
    def pr(f,c,R):
        d=D@R.T;u,v,th=proje(d[...,0],d[...,1],d[...,2],c);val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        return cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0)),val,th
    e1,v1,t1=pr(f1,c1,ROLL);e2,v2,t2=pr(f2,c2,R12@ROLL);u1=v1&(~v2|(t1<=t2));u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1);comb[u1]=e1[u1];comb[u2]=e2[u2];return comb
def img(fid,cam):
    return cv2.imread(f"{FIR}/{cam}/"+[f for f in os.listdir(f'{FIR}/{cam}') if f'_{fid}_' in f][0])

def draw_axis(im,R,t,cam,O,axw,L,color,label):
    pts=[]
    for tt in np.linspace(0,1,40):
        Pc=R@(O+tt*L*axw)+t
        if Pc[2]<=0.03: continue
        u,v,th=projf(Pc,cam)
        if th>np.deg2rad(FOV/2) or not(0<=u<3264 and 0<=v<3264): continue
        pts.append((int(round(u)),int(round(v))))
    for i in range(1,len(pts)): cv2.line(im,pts[i-1],pts[i],color,10)
    if len(pts)>=2:
        cv2.arrowedLine(im,pts[-2],pts[-1],color,10,tipLength=2.0)
        cv2.putText(im,label,(pts[-1][0]+12,pts[-1][1]),cv2.FONT_HERSHEY_SIMPLEX,2.4,color,6)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fid",default="729")
    ap.add_argument("--D",type=float,default=3.0); ap.add_argument("--L",type=float,default=1.5); a=ap.parse_args()
    P1,P2=load_pose_t(); R12=R12_from(P1,P2); ld=json.load(open(LINES)); c1,c2=ld['cam1'],ld['cam2']
    R1,t1=P1[a.fid]; R2,t2=P2[a.fid]
    C1=-R1.T@t1; C2=-R2.T@t2
    f1=img(a.fid,'cam1').copy(); f2=img(a.fid,'cam2').copy()
    # BGR colors
    AX=[('X',np.array([1.,0,0]),(0,0,255)),('Y',np.array([0,1.,0]),(0,220,0)),('Z',np.array([0,0,1.]),(255,120,0))]
    # anchor each gizmo in front of its own lens (optical axis in world = R^T [0,0,1])
    O1=C1+a.D*(R1.T@np.array([0,0,1.])); O2=C2+a.D*(R2.T@np.array([0,0,1.]))
    for nm,w,col in AX:
        draw_axis(f1,R1,t1,c1,O1,w,a.L,col,f"+{nm}")
        draw_axis(f2,R2,t2,c2,O2,w,a.L,col,f"+{nm}")
    eq=render(f1,f2,c1,c2,R12,EQW,EQH)
    fig,ax=plt.subplots(figsize=(22,11)); ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
    # --- horizon circle (world horizontal plane, perp to world-up +Y) in panorama frame ---
    Rp=ROLL@R1
    horiz=[Rp@np.array([np.cos(t),0,np.sin(t)]) for t in np.linspace(0,2*np.pi,721)]
    draw_curve(ax,horiz,EQW,EQH,'yellow',lw=2,alpha=0.85)
    # cardinal azimuth ticks: world +X,+Z,-X,-Z on the horizon
    for lbl,wd,col in [('+X',[1,0,0],'red'),('+Z',[0,0,1],'deepskyblue'),('-X',[-1,0,0],'red'),('-Z',[0,0,-1],'deepskyblue')]:
        d=Rp@np.array(wd,float); x,y=dpx1(d,EQW,EQH)
        ax.scatter([x],[y],s=120,facecolors='none',edgecolors=col,linewidths=2.5,zorder=7)
        ax.text(x,y-14,lbl,color=col,fontsize=14,weight='bold',ha='center',zorder=8,
                bbox=dict(fc='black',alpha=.4,pad=1))
    ax.set_title(f"END-TO-END gizmos + horizon (yellow=world horizontal plane; ticks=world azimuths)  frame {a.fid}",fontsize=13)
    ax.axis('off'); fig.tight_layout(); fig.savefig(f"{OUT}/gizmo_e2e_{a.fid}.png",dpi=120,bbox_inches='tight'); plt.close(fig)
    print(f"WROTE {OUT}/gizmo_e2e_{a.fid}.png  (D={a.D}, L={a.L})")

if __name__=="__main__":
    main()
