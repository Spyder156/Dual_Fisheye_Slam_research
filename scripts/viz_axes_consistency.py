#!/usr/bin/env python3
"""
Global coordinate-consistency check.
Render two different frames' equirects and overlay the WORLD axis triad
(X=red, Y=green, Z=blue; +axis solid dot+label, -axis hollow) using each
frame's camera pose. If poses + our convention are globally consistent, the
SAME world axis lands on the SAME physical thing in both (e.g. world-up axis
points at the ceiling in both).
Output -> outputs/armC/investigate/axes_consistency.png
"""
import os, re, json, numpy as np, cv2
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
    """id -> R1 (world->cam1) and R2 (world->cam2)."""
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
def dpx(d,W,H):
    X,Y,Z=d[0],d[1],d[2];lon=np.arctan2(X,Z);lat=np.arcsin(np.clip(-Y,-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H
def render(f1,f2,c1,c2,R12,W,H):
    D=dirs(W,H)
    def pr(f,c,R):
        d=D@R.T;u,v,th=proj(d[...,0],d[...,1],d[...,2],c);val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        return cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0)),val,th
    e1,v1,t1=pr(f1,c1,ROLL);e2,v2,t2=pr(f2,c2,R12@ROLL);u1=v1&(~v2|(t1<=t2));u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1);comb[u1]=e1[u1];comb[u2]=e2[u2];return comb

def frame_img(fid,cam):
    return cv2.imread(f"{FIR}/{cam}/"+[f for f in os.listdir(f'{FIR}/{cam}') if f'_{fid}_' in f][0])

def main():
    R1s,R2s=load_poses(); R12=R12_from(R1s,R2s)
    ld=json.load(open(LINES)); c1,c2=ld['cam1'],ld['cam2']
    # pick frame A=729, B = the one with the largest rotation vs A (so poses really differ)
    A='729'; RA=R1s[A]
    cand={k:np.degrees(np.arccos(np.clip((np.trace(RA@R1s[k].T)-1)/2,-1,1))) for k in R1s if k!=A}
    B=max(cand,key=cand.get)
    print(f"frame A={A}, frame B={B} (pose differs by {cand[B]:.1f} deg)")

    axes=[('+X',np.array([1,0,0]),(255,0,0),True),('-X',np.array([-1,0,0]),(255,0,0),False),
          ('+Y',np.array([0,1,0]),(0,200,0),True),('-Y',np.array([0,-1,0]),(0,200,0),False),
          ('+Z',np.array([0,0,1]),(0,0,255),True),('-Z',np.array([0,0,-1]),(0,0,255),False)]

    fig,ax=plt.subplots(2,1,figsize=(20,20))
    for pi,fid in enumerate([A,B]):
        eq=render(frame_img(fid,'cam1'),frame_img(fid,'cam2'),c1,c2,R12,EQW,EQH)
        R1=R1s[fid]
        ax[pi].imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
        for name,w,col,solid in axes:
            d_eq=ROLL@(R1@w)                       # world axis -> equirect frame
            x,y=dpx(d_eq,EQW,EQH); c=np.array(col)/255
            if solid:
                ax[pi].scatter([x],[y],s=260,c=[c],edgecolors='white',linewidths=1.5,zorder=5)
                ax[pi].text(x+14,y-14,name,color='white',fontsize=15,weight='bold',
                            bbox=dict(fc=tuple(c),alpha=.8,pad=1),zorder=6)
            else:
                ax[pi].scatter([x],[y],s=160,facecolors='none',edgecolors=[c],linewidths=2.5,zorder=5)
                ax[pi].text(x+10,y+18,name,color=tuple(c),fontsize=11,zorder=6)
        ax[pi].set_title(f"frame {fid}   (world X=red Y=green Z=blue; +solid  -hollow)",fontsize=14)
        ax[pi].axis('off')
    fig.suptitle("Global coordinate consistency — same world axis should hit the same physical thing in both",fontsize=15)
    fig.tight_layout();fig.savefig(f"{OUT}/axes_consistency.png",dpi=120,bbox_inches='tight');plt.close(fig)
    print(f"WROTE {OUT}/axes_consistency.png")

if __name__=="__main__":
    main()
