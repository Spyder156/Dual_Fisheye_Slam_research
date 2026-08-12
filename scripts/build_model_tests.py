#!/usr/bin/env python3
"""
Arm C — per-experiment model tests, each fully self-contained.

For every camera-model hypothesis:
  1. fit it to the manual overlap correspondences (ray-consistency objective)
  2. RE-RENDER the equirect (hard seam, no blend) with THAT config's fitted
     intrinsics + extrinsics
  3. overlay the 7 correspondences on its OWN image, fine markers:
       red '+'  = where cam1's ray lands   cyan 'x' = where cam2's ray lands
       yellow line = residual (should shrink; both should sit on the real feature)
  Output -> outputs/armC/model_tests/<config>/  (equirect.png, overlay.png, metrics.txt)
Plus a _summary/ with the train-vs-LOO comparison chart.

KB-N: theta_d = theta*(1 + sum_{j=1..N} k_j theta^{2j}).
"""
import os, re, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
PAIRS = f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
MT = f"{ROOT}/outputs/armC/model_tests"
ROLL = Rot.from_euler('z', 180, degrees=True).as_matrix()
FID = "729"; EQW = 4096; EQH = 2048; PXDEG = EQW/360.0; FOV=200.0; REG=5e-3


# ---------- geometry ----------
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

def kb_unproject(u,v,fx,fy,cx,cy,k):
    xp=(u-cx)/fx; yp=(v-cy)/fy; thd=np.hypot(xp,yp); th=thd.copy(); N=len(k)
    for _ in range(15):
        poly=np.ones_like(th); dpoly=np.zeros_like(th)
        for j in range(1,N+1):
            poly=poly+k[j-1]*th**(2*j); dpoly=dpoly+k[j-1]*(2*j)*th**(2*j-1)
        th=th-(th*poly-thd)/(poly+th*dpoly)
    phi=np.arctan2(yp,xp)
    return np.stack([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)],-1)

def kb_project(X,Y,Z,fx,fy,cx,cy,k):
    r=np.sqrt(X*X+Y*Y); th=np.arctan2(r,Z); poly=np.ones_like(th)
    for j in range(1,len(k)+1): poly=poly+k[j-1]*th**(2*j)
    td=th*poly; s=np.where(r>1e-9,td/np.where(r>1e-9,r,1),0.0)
    return fx*X*s+cx, fy*Y*s+cy, th

def equirect_dirs(W,H):
    lon=(np.arange(W)+0.5)/W*2*np.pi-np.pi; lat=np.pi/2-(np.arange(H)+0.5)/H*np.pi
    lon,lat=np.meshgrid(lon,lat)
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],-1)

def dir_to_px(d,W,H):
    X,Y,Z=d[...,0],d[...,1],d[...,2]
    lon=np.arctan2(X,Z); lat=np.arcsin(np.clip(-Y,-1,1))
    return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H


# ---------- model params ----------
def cam_dict(base,N):
    k=list(base["k"])+[0.0]*(N-len(base["k"]))
    return dict(fx=base["fx"],fy=base["fy"],cx=base["cx"],cy=base["cy"],k=k)

def flat(cam): return np.array([cam["fx"],cam["fy"]]+list(cam["k"]))
def unflat(x,cx,cy): return dict(fx=x[0],fy=x[1],cx=cx,cy=cy,k=list(x[2:]))


# ---------- fit ----------
def angles(r1,r2,R12):
    d=np.clip((r1*(r2@R12)).sum(1),-1,1); return np.degrees(np.arccos(d))

def rays(P,c1,c2):
    r1=kb_unproject(P[:,0],P[:,1],c1["fx"],c1["fy"],c1["cx"],c1["cy"],c1["k"])
    r2=kb_unproject(P[:,2],P[:,3],c2["fx"],c2["fy"],c2["cx"],c2["cy"],c2["k"])
    return r1,r2

def fit(P,cfg,c1_0,c2_0,R0):
    c1p,c2p=flat(c1_0),flat(c2_0)
    x0=[];
    if cfg["rot"]: x0+=[0,0,0]
    if cfg["intr"]: x0+=list(c1p)+list(c2p)
    x0=np.array(x0,float)
    def split(x):
        i=0; rv=x[i:i+3] if cfg["rot"] else np.zeros(3); i+= 3 if cfg["rot"] else 0
        R12=Rot.from_rotvec(rv).as_matrix()@R0
        if cfg["intr"]:
            n=len(c1p); c1=unflat(x[i:i+n],c1_0["cx"],c1_0["cy"]); c2=unflat(x[i+n:i+2*n],c2_0["cx"],c2_0["cy"])
        else: c1,c2=c1_0,c2_0
        return R12,c1,c2
    if len(x0)==0:
        R12,c1,c2=R0,c1_0,c2_0
    else:
        def res(x):
            R12,c1,c2=split(x); r1,r2=rays(P,c1,c2); e=(r1-r2@R12).ravel()
            if cfg["intr"]: e=np.concatenate([e,REG*np.concatenate([flat(c1)-c1p,flat(c2)-c2p])])
            return e
        s=least_squares(res,x0,method='lm',max_nfev=5000); R12,c1,c2=split(s.x)
    r1,r2=rays(P,c1,c2)
    return R12,c1,c2,angles(r1,r2,R12)

def loo(P,cfg,c1_0,c2_0,R0):
    o=[]
    for i in range(len(P)):
        m=np.ones(len(P),bool); m[i]=False
        R12,c1,c2,_=fit(P[m],cfg,c1_0,c2_0,R0)
        r1,r2=rays(P[i:i+1],c1,c2); o.append(angles(r1,r2,R12)[0])
    return np.array(o)


# ---------- render ----------
def render_hardseam(f1,f2,c1,c2,R12,W,H):
    dirs=equirect_dirs(W,H)
    def proj(fish,cam,R):
        d=dirs@R.T; u,v,th=kb_project(d[...,0],d[...,1],d[...,2],cam["fx"],cam["fy"],cam["cx"],cam["cy"],cam["k"])
        val=(th<=np.deg2rad(FOV/2))&(u>=0)&(u<3264)&(v>=0)&(v<3264)
        eq=cv2.remap(fish,np.where(val,u,-1).astype(np.float32),np.where(val,v,-1).astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
        return eq,val,th
    eq1,v1,t1=proj(f1,c1,ROLL); eq2,v2,t2=proj(f2,c2,R12@ROLL)
    use1=v1&(~v2|(t1<=t2)); use2=v2&(~v1|(t2<t1))
    comb=np.zeros_like(eq1); comb[use1]=eq1[use1]; comb[use2]=eq2[use2]
    return comb


def overlay(eq,P,c1,c2,R12,path,title):
    r1,r2=rays(P,c1,c2); H,W=eq.shape[:2]
    d1=r1@ROLL.T; d2=(r2@R12)@ROLL.T
    x1,y1=dir_to_px(d1,W,H); x2,y2=dir_to_px(d2,W,H)
    fig,ax=plt.subplots(figsize=(22,11)); ax.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
    for i in range(len(P)):
        ax.plot([x1[i],x2[i]],[y1[i],y2[i]],'-',c='yellow',lw=0.6)
        ax.plot(x1[i],y1[i],'+',c='red',ms=7,mew=0.8)
        ax.plot(x2[i],y2[i],'x',c='cyan',ms=5,mew=0.8)
    ax.set_title(title,fontsize=13); ax.axis('off')
    fig.tight_layout(); fig.savefig(path,dpi=140,bbox_inches='tight'); plt.close(fig)


def main():
    d=json.load(open(PAIRS))
    P=np.array([[p["u1"],p["v1"],p["u2"],p["v2"]] for p in d["pairs"]])
    R0=recover_R12()
    f1=cv2.imread(f"{FIR}/cam1/"+[f for f in os.listdir(f'{FIR}/cam1') if f'_{FID}_' in f][0])
    f2=cv2.imread(f"{FIR}/cam2/"+[f for f in os.listdir(f'{FIR}/cam2') if f'_{FID}_' in f][0])

    configs=[("baseline_kb4",4,dict(rot=False,intr=False)),
             ("kb4_extr_rot",4,dict(rot=True,intr=False)),
             ("kb4_intr_extr",4,dict(rot=True,intr=True)),
             ("kb6_intr_extr",6,dict(rot=True,intr=True)),
             ("kb8_intr_extr",8,dict(rot=True,intr=True))]
    rows=[]
    for name,N,cfg in configs:
        c1_0,c2_0=cam_dict(d["cam1"],N),cam_dict(d["cam2"],N)
        R12,c1,c2,ang=fit(P,cfg,c1_0,c2_0,R0)
        lo=loo(P,cfg,c1_0,c2_0,R0) if (cfg["rot"] or cfg["intr"]) else ang
        folder=f"{MT}/{name}"; os.makedirs(folder,exist_ok=True)
        eq=render_hardseam(f1,f2,c1,c2,R12,EQW,EQH)
        cv2.imwrite(f"{folder}/equirect_{FID}.png",eq)
        overlay(eq,P,c1,c2,R12,f"{folder}/overlay_{FID}.png",
                f"{name}   median disagreement {np.median(ang):.3f} deg ({np.median(ang)*PXDEG:.1f} eq-px)   LOO {np.median(lo):.3f} deg")
        with open(f"{folder}/metrics.txt","w") as fh:
            fh.write(f"config={name} N={N} fit_rot={cfg['rot']} fit_intr={cfg['intr']}\n")
            fh.write(f"train_median_deg={np.median(ang):.4f} train_max_deg={ang.max():.4f} eqpx={np.median(ang)*PXDEG:.2f}\n")
            fh.write(f"LOO_median_deg={np.median(lo):.4f} LOO_mean_deg={lo.mean():.4f}\n")
            fh.write(f"per_pair_deg={np.round(ang,3).tolist()}\n")
            fh.write(f"R12=\n{R12}\ncam1={c1}\ncam2={c2}\n")
        rows.append((name,np.median(ang),np.median(lo)))
        print(f"{name:16s} train_med={np.median(ang):.3f}deg LOO_med={np.median(lo):.3f}deg -> {folder}/")

    os.makedirs(f"{MT}/_summary",exist_ok=True)
    names=[r[0] for r in rows]; tr=[r[1] for r in rows]; lo=[r[2] for r in rows]; x=np.arange(len(names))
    plt.figure(figsize=(11,5))
    plt.bar(x-0.2,tr,0.4,label='train median',color='steelblue')
    plt.bar(x+0.2,lo,0.4,label='leave-one-out median',color='indianred')
    for i,(t,l) in enumerate(zip(tr,lo)):
        plt.text(i-0.2,t,f'{t:.2f}',ha='center',va='bottom',fontsize=8); plt.text(i+0.2,l,f'{l:.2f}',ha='center',va='bottom',fontsize=8)
    plt.xticks(x,names,rotation=15); plt.ylabel('overlap ray disagreement (deg)')
    plt.title(f'Model comparison — {len(P)} correspondences'); plt.legend(); plt.tight_layout()
    plt.savefig(f"{MT}/_summary/model_comparison.png",dpi=120); plt.close()
    print(f"WROTE per-config folders + {MT}/_summary/model_comparison.png")


if __name__=="__main__":
    main()
