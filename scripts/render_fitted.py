#!/usr/bin/env python3
"""
Render the fitted lens model (outputs/armC/lens_fit/fitted_params.json) to a
hard-seam equirect, and overlay the overlap correspondences (should coincide)
on it. This is the physical check: doubles gone, no beautifying.
Also renders the BASELINE (FIORD params) at the same size for side-by-side.
Outputs -> outputs/armC/lens_fit/
"""
import os, re, json, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
PAIRS = f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
OUT = f"{ROOT}/outputs/armC/lens_fit"; FID="729"; EQW=4096; EQH=2048; FOV=200.0
ROLL = np.array([[np.cos(np.pi),-np.sin(np.pi),0],[np.sin(np.pi),np.cos(np.pi),0],[0,0,1]])

def qR(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def recover_R12():
    R1,R2={},{}
    L=[l for l in open(BTXT) if not l.startswith("#") and l.strip()]
    for i in range(0,len(L),2):
        p=L[i].split(); m=re.search(r'_(\d+)_fisheye(\d)',p[9])
        if m:(R1 if m.group(2)=='1' else R2)[m.group(1)]=qR(*map(float,p[1:5]))
    rels=np.array([R2[k]@R1[k].T for k in set(R1)&set(R2)])
    U,_,Vt=np.linalg.svd(rels.mean(0)); R=U@Vt
    if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
    return R

def kb_project(X,Y,Z,c):
    r=np.sqrt(X*X+Y*Y); th=np.arctan2(r,Z); poly=np.ones_like(th)
    for j in range(1,len(c["k"])+1): poly=poly+c["k"][j-1]*th**(2*j)
    s=np.where(r>1e-9,th*poly/np.where(r>1e-9,r,1),0.0)
    return c["fx"]*X*s+c["cx"], c["fy"]*Y*s+c["cy"], th
def kb_unproject(u,v,c):
    xp=(u-c["cx"])/c["fx"]; yp=(v-c["cy"])/c["fy"]; thd=np.hypot(xp,yp); th=thd.copy()
    for _ in range(18):
        poly=np.ones_like(th); dpoly=np.zeros_like(th)
        for j in range(1,len(c["k"])+1): poly=poly+c["k"][j-1]*th**(2*j); dpoly=dpoly+c["k"][j-1]*(2*j)*th**(2*j-1)
        th=th-(th*poly-thd)/(poly+th*dpoly)
    phi=np.arctan2(yp,xp)
    return np.stack([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)],-1)

def dirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi; lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi
    lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
def dpx(d,W,H):
    X,Y,Z=d[...,0],d[...,1],d[...,2]; lon=np.arctan2(X,Z); lat=np.arcsin(np.clip(-Y,-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H

def render(f1,f2,c1,c2,R12,W,H):
    D=dirs(W,H)
    def proj(f,c,R):
        d=D@R.T; u,v,th=kb_project(d[...,0],d[...,1],d[...,2],c)
        val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        eq=cv2.remap(f,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        return eq,val,th
    e1,v1,t1=proj(f1,c1,ROLL); e2,v2,t2=proj(f2,c2,R12@ROLL)
    u1=v1&(~v2|(t1<=t2)); u2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(e1); comb[u1]=e1[u1]; comb[u2]=e2[u2]
    return comb

def main():
    fp=json.load(open(f"{OUT}/fitted_params.json")); pd=json.load(open(PAIRS))
    c1,c2=fp["cam1"],fp["cam2"]; R12=np.array(fp["R12"])
    f1=cv2.imread(f"{FIR}/cam1/"+[f for f in os.listdir(f'{FIR}/cam1') if f'_{FID}_' in f][0])
    f2=cv2.imread(f"{FIR}/cam2/"+[f for f in os.listdir(f'{FIR}/cam2') if f'_{FID}_' in f][0])
    ld=json.load(open(f"{ROOT}/outputs/armC/manual_lines/lines_729.json"))
    base1,base2=ld["cam1"],ld["cam2"]; R0=recover_R12()

    eqf=render(f1,f2,c1,c2,R12,EQW,EQH); cv2.imwrite(f"{OUT}/equirect_fitted.png",eqf)
    eqb=render(f1,f2,base1,base2,R0,EQW,EQH); cv2.imwrite(f"{OUT}/equirect_baseline.png",eqb)

    P=np.array([[p["u1"],p["v1"],p["u2"],p["v2"]] for p in pd["pairs"]])
    def overlay(eq,c1,c2,R12,path,title):
        r1=kb_unproject(P[:,0],P[:,1],c1); r2=kb_unproject(P[:,2],P[:,3],c2)
        H,W=eq.shape[:2]; d1=r1@ROLL.T; d2=(r2@R12)@ROLL.T
        x1,y1=dpx(d1,W,H); x2,y2=dpx(d2,W,H)
        fig,ax=plt.subplots(figsize=(22,11)); ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
        for i in range(len(P)):
            ax.plot([x1[i],x2[i]],[y1[i],y2[i]],'-',c='yellow',lw=0.6)
            ax.plot(x1[i],y1[i],'+',c='red',ms=7,mew=0.8); ax.plot(x2[i],y2[i],'x',c='cyan',ms=5,mew=0.8)
        ax.set_title(title,fontsize=13); ax.axis('off'); fig.tight_layout()
        fig.savefig(path,dpi=140,bbox_inches='tight'); plt.close(fig)
    overlay(eqf,c1,c2,R12,f"{OUT}/overlay_fitted.png",f"FITTED  line={fp['rms_line_deg']:.3f}deg overlap={fp['rms_overlap_deg']:.3f}deg")
    overlay(eqb,base1,base2,R0,f"{OUT}/overlay_baseline.png",f"BASELINE  line={fp['base_line_deg']:.3f}deg overlap={fp['base_overlap_deg']:.3f}deg")
    print(f"WROTE {OUT}/equirect_fitted.png, equirect_baseline.png, overlay_fitted.png, overlay_baseline.png")

if __name__=="__main__":
    main()
