#!/usr/bin/env python3
"""
Arm C — compare camera-model hypotheses on the manual overlap correspondences.

Residual = angle between the two lenses' bearing rays to the same world point
(cam2 ray brought into rig frame via R12). We fit several configs jointly and
report TRAIN + leave-one-out (LOO) angular disagreement:

  baseline KB4 | KB4+extrinsic | KB4 intr+extrinsic | KB6+extr | KB8+extr

KB-N radial model:  theta_d = theta * (1 + sum_{j=1..N} k_j theta^{2j}).
Extrinsic = R12 rotation (3 DOF). Intrinsics fitted = per-cam fx,fy + k[1..N],
lightly regularized toward the FIORD priors (7 pairs is thin). cx,cy fixed.
Baseline translation (parallax) is a SEPARATE later test (needs epipolar/depth).

Outputs -> outputs/armC/model_tests/_fit/
"""
import os, re, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
PAIRS = f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
BASE_EQ = f"{ROOT}/outputs/armC/model_tests/baseline_kb/equirect_729.png"
OUT = f"{ROOT}/outputs/armC/model_tests/_fit"; os.makedirs(OUT, exist_ok=True)
ROLL = Rot.from_euler('z', 180, degrees=True).as_matrix()
EQW = 5760; PXDEG = EQW/360.0
REG = 5e-3   # Tikhonov weight on intrinsic deviation from prior


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
    xp=(u-cx)/fx; yp=(v-cy)/fy; thd=np.hypot(xp,yp); th=thd.copy()
    N=len(k)
    for _ in range(15):
        t2=th*th; poly=np.ones_like(th); dpoly=np.zeros_like(th); tp=t2.copy()
        f=th.copy()
        # theta_d = th*(1+sum k_j th^{2j})
        s=np.ones_like(th); ds=np.zeros_like(th); p=np.ones_like(th)
        acc=np.ones_like(th); dacc=np.zeros_like(th)
        val=np.ones_like(th)
        # compute poly=1+sum k_j th^{2j}, dpoly/dth = sum k_j*2j*th^{2j-1}
        poly=np.ones_like(th); dpoly=np.zeros_like(th)
        for j in range(1,N+1):
            poly=poly+k[j-1]*th**(2*j)
            dpoly=dpoly+k[j-1]*(2*j)*th**(2*j-1)
        f=th*poly-thd; df=poly+th*dpoly
        th=th-f/df
    phi=np.arctan2(yp,xp)
    return np.stack([np.sin(th)*np.cos(phi),np.sin(th)*np.sin(phi),np.cos(th)],-1)

def angles(r1,r2,R12):
    d=np.clip((r1*(r2@R12)).sum(1),-1,1); return np.degrees(np.arccos(d))


class Cfg:
    def __init__(s,name,N,fit_rot,fit_intr): s.name,s.N,s.fit_rot,s.fit_intr=name,N,fit_rot,fit_intr

def prior_intr(cam,N):
    k=list(cam["k"])+[0.0]*(N-len(cam["k"])); return np.array([cam["fx"],cam["fy"]]+k)

def unpack(x,cfg,c1p,c2p,R0):
    i=0
    if cfg.fit_rot: rv=x[i:i+3]; i+=3
    else: rv=np.zeros(3)
    R12=Rot.from_rotvec(rv).as_matrix()@R0
    if cfg.fit_intr:
        n=2+cfg.N; a=x[i:i+n]; b=x[i+n:i+2*n]; i+=2*n
    else:
        a,b=c1p,c2p
    return R12,a,b

def rays_from(P,a,b,cx,cy):
    r1=kb_unproject(P[:,0],P[:,1],a[0],a[1],cx,cy,a[2:])
    r2=kb_unproject(P[:,2],P[:,3],b[0],b[1],cx,cy,b[2:])
    return r1,r2

def fit(P,cfg,cam1,cam2,R0):
    c1p=prior_intr(cam1,cfg.N); c2p=prior_intr(cam2,cfg.N)
    x0=[]
    if cfg.fit_rot: x0+=[0,0,0]
    if cfg.fit_intr: x0+=list(c1p)+list(c2p)
    x0=np.array(x0) if x0 else np.zeros(0)
    def res(x):
        R12,a,b=unpack(x,cfg,c1p,c2p,R0)
        r1,r2=rays_from(P,a,b,cam1["cx"],cam1["cy"])
        e=(r1-r2@R12).ravel()
        if cfg.fit_intr:
            reg=REG*np.concatenate([(a-c1p),(b-c2p)])
            e=np.concatenate([e,reg])
        return e
    if len(x0)==0:
        R12,a,b=R0,c1p,c2p
    else:
        s=least_squares(res,x0,method='lm',max_nfev=4000)
        R12,a,b=unpack(s.x,cfg,c1p,c2p,R0)
    r1,r2=rays_from(P,a,b,cam1["cx"],cam1["cy"])
    return R12,a,b,angles(r1,r2,R12)

def loo(P,cfg,cam1,cam2,R0):
    outs=[]
    for i in range(len(P)):
        m=np.ones(len(P),bool); m[i]=False
        R12,a,b,_=fit(P[m],cfg,cam1,cam2,R0)
        r1,r2=rays_from(P[i:i+1],a,b,cam1["cx"],cam1["cy"])
        outs.append(angles(r1,r2,R12)[0])
    return np.array(outs)


def main():
    d=json.load(open(PAIRS)); cam1,cam2=d["cam1"],d["cam2"]
    P=np.array([[p["u1"],p["v1"],p["u2"],p["v2"]] for p in d["pairs"]])
    R0=recover_R12()
    cfgs=[Cfg("baseline KB4",4,False,False),
          Cfg("KB4 + extr",4,True,False),
          Cfg("KB4 intr+extr",4,True,True),
          Cfg("KB6 + extr",6,True,True),
          Cfg("KB8 + extr",8,True,True)]
    rows=[]; best=None
    for c in cfgs:
        R12,a,b,ang=fit(P,c,cam1,cam2,R0)
        lo=loo(P,c,cam1,cam2,R0) if (c.fit_rot or c.fit_intr) else ang
        rows.append((c.name,np.median(ang),ang.max(),np.median(lo)))
        print(f"{c.name:16s}: train med={np.median(ang):.3f}deg ({np.median(ang)*PXDEG:.1f}px) max={ang.max():.3f} | LOO med={np.median(lo):.3f}deg")
        if best is None or np.median(ang)<best[1]: best=(c,np.median(ang),R12,a,b,ang)

    # chart: train vs LOO per config
    names=[r[0] for r in rows]; tr=[r[1] for r in rows]; lo=[r[3] for r in rows]
    x=np.arange(len(names))
    plt.figure(figsize=(11,5))
    plt.bar(x-0.2,tr,0.4,label='train median',color='steelblue')
    plt.bar(x+0.2,lo,0.4,label='leave-one-out median',color='indianred')
    for i,(t,l) in enumerate(zip(tr,lo)):
        plt.text(i-0.2,t,f'{t:.2f}',ha='center',va='bottom',fontsize=8)
        plt.text(i+0.2,l,f'{l:.2f}',ha='center',va='bottom',fontsize=8)
    plt.xticks(x,names,rotation=15); plt.ylabel('overlap ray disagreement (deg)')
    plt.title('Camera-model comparison — overlap disagreement (7 pairs)')
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison.png",dpi=120); plt.close()

    # overlay: baseline vs best on equirect
    eq=cv2.imread(BASE_EQ); H,W=eq.shape[:2]
    def eqpx(ray):
        dd=ray@ROLL.T; X,Y,Z=dd[...,0],dd[...,1],dd[...,2]
        lon=np.arctan2(X,Z); lat=np.arcsin(np.clip(-Y,-1,1))
        return (lon+np.pi)/(2*np.pi)*W,(np.pi/2-lat)/np.pi*H
    r1b,r2b=rays_from(P,prior_intr(cam1,4),prior_intr(cam2,4),cam1["cx"],cam1["cy"])
    cbest,_,R12b,ab,bb,angb=best
    r1o,r2o=rays_from(P,ab,bb,cam1["cx"],cam1["cy"])
    fig,ax=plt.subplots(2,1,figsize=(20,12))
    for axi,(r1,r2,R12,tt) in zip(ax,[(r1b,r2b,R0,f"baseline KB4  median {np.median(angles(r1b,r2b,R0)):.2f} deg"),
                                       (r1o,r2o,R12b,f"BEST: {cbest.name}  median {np.median(angb):.2f} deg")]):
        axi.imshow(cv2.cvtColor(eq,cv2.COLOR_BGR2RGB))
        x1,y1=eqpx(r1); x2,y2=eqpx(r2@R12)
        for i in range(len(r1)):
            axi.plot([x1[i],x2[i]],[y1[i],y2[i]],'-',c='yellow',lw=1.3)
            axi.scatter([x1[i]],[y1[i]],c='red',s=28,zorder=3); axi.scatter([x2[i]],[y2[i]],c='cyan',s=28,zorder=3)
        axi.set_title(tt); axi.axis('off')
    fig.tight_layout(); fig.savefig(f"{OUT}/baseline_vs_best.png",dpi=110,bbox_inches='tight'); plt.close(fig)
    print(f"WROTE {OUT}/model_comparison.png , baseline_vs_best.png")


if __name__=="__main__":
    main()
