#!/usr/bin/env python3
"""
Two world-axis gizmos on ONE equirect, one at each lens's optical center
(cam1 at panorama center, cam2 at the back). Both draw the SAME world axes
(X=red,Y=green,Z=blue). If poses+convention are globally consistent, both
gizmos point the same world directions in the scene (Y up in both), and each
axis is SOLID in the lens whose hemisphere it points into, DASHED in the other.
Arrows are great-circle arcs (seam-aware). Output -> outputs/armC/investigate/two_triax_<fid>.png
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
def gdirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi;lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi;lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
def dpx1(d,W,H):
    d=d/np.linalg.norm(d);lon=np.arctan2(d[0],d[2]);lat=np.arcsin(np.clip(-d[1],-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H
def render(f1,f2,c1,c2,R12,W,H):
    D=gdirs(W,H)
    def pr(f,c,R):
        d=D@R.T;u,v,th=proj(d[...,0],d[...,1],d[...,2],c);val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        return cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0)),val,th
    e1,v1,t1=pr(f1,c1,ROLL);e2,v2,t2=pr(f2,c2,R12@ROLL);u1=v1&(~v2|(t1<=t2));u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1);comb[u1]=e1[u1];comb[u2]=e2[u2];return comb
def img(fid,cam):
    return cv2.imread(f"{FIR}/{cam}/"+[f for f in os.listdir(f'{FIR}/{cam}') if f'_{fid}_' in f][0])

def arc(ax,c,a,col,solid,W,H,s=0.5,label=""):
    ts=np.linspace(0,1,26); prev=None; xs=[];ys=[]
    for t in ts:
        d=c+t*s*a; x,y=dpx1(d,W,H)
        if prev is not None and abs(x-prev)>W/2:      # seam wrap -> break
            ax.plot(xs,ys,color=col,lw=3.2,alpha=1 if solid else .45,ls='-' if solid else (0,(4,3)),zorder=7);xs=[];ys=[]
        xs.append(x);ys.append(y);prev=x
    ax.plot(xs,ys,color=col,lw=3.2,alpha=1 if solid else .45,ls='-' if solid else (0,(4,3)),zorder=7)
    ax.scatter([xs[-1]],[ys[-1]],s=70,c=col,marker=(3,0,np.degrees(np.arctan2(ys[-1]-ys[-2],xs[-1]-xs[-2]))-90),zorder=8,alpha=1 if solid else .5)
    ax.text(xs[-1]+6,ys[-1]-6,label,color=col,fontsize=14,weight='bold',zorder=9)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fid",default="729"); a=ap.parse_args()
    R1s,R2s=load_poses(); R12=R12_from(R1s,R2s); ld=json.load(open(LINES)); c1,c2=ld['cam1'],ld['cam2']
    eq=render(img(a.fid,'cam1'),img(a.fid,'cam2'),c1,c2,R12,EQW,EQH)
    R1=R1s[a.fid]
    axes=[('X',np.array([1.,0,0]),'red'),('Y',np.array([0,1.,0]),'lime'),('Z',np.array([0,0,1.]),'deepskyblue')]
    Aw=[(nm,(ROLL@(R1@w)),col) for nm,w,col in axes]           # world axes in panorama frame
    cen1=np.array([0,0,1.])                                     # cam1 optical center (panorama center)
    cen2=ROLL@(R12.T@np.array([0,0,1.])); cen2/=np.linalg.norm(cen2)  # cam2 center in panorama

    fig,ax=plt.subplots(figsize=(22,11)); ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
    for cen,fwd,tag in [(cen1,cen1,"cam1"),(cen2,cen2,"cam2")]:
        px,py=dpx1(cen,EQW,EQH); ax.scatter([px],[py],s=90,c='white',edgecolors='k',zorder=9)
        ax.text(px,py-22,tag,color='white',fontsize=13,weight='bold',ha='center',zorder=9,
                bbox=dict(fc='black',alpha=.5,pad=1))
        for nm,a_w,col in Aw:
            solid = np.dot(a_w,fwd) >= 0                        # into this lens's hemisphere?
            arc(ax,cen,a_w,col,solid,EQW,EQH,label=f"+{nm}")
    ax.set_title(f"Two world-axis gizmos (cam1 & cam2)  frame {a.fid}   X=red Y=green Z=blue   "
                 f"solid=points into that lens, dashed=behind it",fontsize=13); ax.axis('off')
    fig.tight_layout(); fig.savefig(f"{OUT}/two_triax_{a.fid}.png",dpi=120,bbox_inches='tight'); plt.close(fig)
    # handedness sanity
    M=ROLL@R1; print("det(world->panorama rotation) =",round(np.linalg.det(M),4),"(should be +1 = right-handed)")
    print(f"WROTE {OUT}/two_triax_{a.fid}.png")

if __name__=="__main__":
    main()
