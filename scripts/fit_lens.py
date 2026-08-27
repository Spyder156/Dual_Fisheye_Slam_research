#!/usr/bin/env python3
"""
Arm C — physical lens fitter (autograd / gradient descent).

Model per lens: Kannala-Brandt with N radial terms, theta_d = theta*(1+sum k_j theta^{2j}).
Initialized AT the FIORD base model (base k's for j<=4, higher orders = 0), so it
starts exactly as the given camera. Extrinsic R12 refined from the recovered value.
We then optimize ALL params (focal, center, radial k's per lens, extrinsic rotation)
by gradient descent to minimize PHYSICAL constraints:

  L_line   : every hand-traced real straight edge must unproject to rays that are
             COPLANAR through the camera center (great circle). Residual = smallest
             eigenvalue of the line's ray-direction scatter (off-plane deviation),
             summed over all lines/points. Constrains the lens at EVERY theta.
  L_overlap: manual cross-lens correspondences must give the SAME world bearing
             (r1 == R12^T r2). Constrains the extrinsic + peripheral agreement.
  L_reg    : light L2 toward the physical prior (only bites where there is no data).

Run (vision env has torch+cv2):
  /home/raghav/miniconda3/envs/vision/bin/python scripts/fit_lens.py --N 6 --steps 4000
Outputs -> outputs/armC/lens_fit/
"""
import os, re, json, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/raghav/workspace/INSV_STITCHING"
FIR = f"{ROOT}/Data/FIORD/MeetingRoom/meetingroom/fisheyeimages"
LINES = f"{ROOT}/outputs/armC/manual_lines/lines_729.json"
PAIRS = f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"
BTXT = f"{ROOT}/outputs/viz/clouds/B_txt/images.txt"
OUT = f"{ROOT}/outputs/armC/lens_fit"; os.makedirs(OUT, exist_ok=True)
FID = "729"; DEV = "cpu"; torch.set_default_dtype(torch.float64)
ROLL = torch.tensor([[np.cos(np.pi), -np.sin(np.pi), 0],
                     [np.sin(np.pi),  np.cos(np.pi), 0], [0, 0, 1]])


# ---------- extrinsic prior ----------
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


def so3_exp(w):
    th=torch.linalg.norm(w)+1e-12
    K=torch.stack([torch.stack([torch.zeros(()),-w[2],w[1]]),
                   torch.stack([w[2],torch.zeros(()),-w[0]]),
                   torch.stack([-w[1],w[0],torch.zeros(())])])
    return torch.eye(3)+torch.sin(th)/th*K+(1-torch.cos(th))/th**2*(K@K)


# ---------- differentiable KB unprojection ----------
def kb_unproject(u,v,fx,fy,cx,cy,k):
    xp=(u-cx)/fx; yp=(v-cy)/fy; thd=torch.sqrt(xp*xp+yp*yp)+1e-12; th=thd.clone()
    for _ in range(18):
        poly=torch.ones_like(th); dpoly=torch.zeros_like(th)
        for j in range(1,len(k)+1):
            poly=poly+k[j-1]*th**(2*j); dpoly=dpoly+k[j-1]*(2*j)*th**(2*j-1)
        th=th-(th*poly-thd)/(poly+th*dpoly)
    phi=torch.atan2(yp,xp)
    return torch.stack([torch.sin(th)*torch.cos(phi),torch.sin(th)*torch.sin(phi),torch.cos(th)],-1)


class Lens(torch.nn.Module):
    def __init__(s,base,N):
        super().__init__()
        s.fx0,s.fy0,s.cx0,s.cy0=base["fx"],base["fy"],base["cx"],base["cy"]
        k0=list(base["k"])+[0.0]*(N-len(base["k"]))
        s.sf=torch.nn.Parameter(torch.zeros(2))      # focal log-scale
        s.dc=torch.nn.Parameter(torch.zeros(2))      # center offset (px)
        s.k =torch.nn.Parameter(torch.tensor(k0))    # radial
        s.k0=torch.tensor(k0)
    def params(s):
        fx=s.fx0*torch.exp(s.sf[0]); fy=s.fy0*torch.exp(s.sf[1])
        return fx,fy,s.cx0+s.dc[0],s.cy0+s.dc[1],s.k
    def unproj(s,u,v):
        fx,fy,cx,cy,k=s.params(); return kb_unproject(u,v,fx,fy,cx,cy,k)


def smallest_plane_eig(D):
    M=D.T@D + 1e-12*torch.eye(3)
    return torch.linalg.eigvalsh(M)[0]           # off-plane scatter (rad^2 * npts)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=6); ap.add_argument("--steps",type=int,default=2500)
    ap.add_argument("--lr",type=float,default=4e-4); ap.add_argument("--w_ov",type=float,default=1.0)
    ap.add_argument("--w_reg",type=float,default=3e-3)
    ap.add_argument("--huber",type=float,default=0.03); a=ap.parse_args()   # rad, robustifies overlap

    ld=json.load(open(LINES)); pd=json.load(open(PAIRS))
    cam1_b,cam2_b=ld["cam1"],ld["cam2"]
    # lines grouped by cam -> tensors
    lines={1:[],2:[]}
    for L in ld["lines"]:
        pts=torch.tensor(L["pts"],dtype=torch.float64)
        lines[L["cam"]].append(pts)
    P=torch.tensor([[p["u1"],p["v1"],p["u2"],p["v2"]] for p in pd["pairs"]])
    R0=torch.tensor(recover_R12())

    L1,L2=Lens(cam1_b,a.N),Lens(cam2_b,a.N)
    drot=torch.nn.Parameter(torch.zeros(3))
    params=list(L1.parameters())+list(L2.parameters())+[drot]
    opt=torch.optim.Adam(params,lr=a.lr)

    def eval_losses():
        # line coplanarity
        ll=torch.zeros(()); npts=0
        for cam,lens in [(1,L1),(2,L2)]:
            for pts in lines[cam]:
                D=lens.unproj(pts[:,0],pts[:,1]); D=D/torch.linalg.norm(D,dim=1,keepdim=True)
                ll=ll+smallest_plane_eig(D); npts+=len(pts)
        R12=so3_exp(drot)@R0
        r1=L1.unproj(P[:,0],P[:,1]); r2=L2.unproj(P[:,2],P[:,3])
        r2r=r2@R12                                  # R12^T r2 (row-vec)
        dist=torch.linalg.norm(r1-r2r,dim=1)        # per-pair chord (~angle rad)
        ov=(dist**2).sum()                          # for reporting (RMS)
        # Huber-robust overlap for optimization (downweights parallax outliers)
        d0=a.huber; ov_rob=torch.where(dist<d0, 0.5*dist**2, d0*(dist-0.5*d0)).sum()
        return ll,npts,ov,ov_rob,R12

    hist=[]
    for it in range(a.steps):
        opt.zero_grad()
        ll,npts,ov,ov_rob,R12=eval_losses()
        reg=(L1.sf**2).sum()+(L2.sf**2).sum()+((L1.dc/50)**2).sum()+((L2.dc/50)**2).sum()
        # penalize deviation of radial from prior, higher orders harder
        wk=torch.tensor([1.0]*4+[8.0]*(a.N-4)) if a.N>4 else torch.ones(a.N)
        reg=reg+((L1.k-L1.k0)**2*wk).sum()+((L2.k-L2.k0)**2*wk).sum()+ (drot**2).sum()
        # straight-line physics is a HARD constraint: dominate the loss so the
        # solver can NEVER bend a real edge to chase overlap (parallax) error.
        loss=1e6*(ll/npts) + a.w_ov*ov_rob/len(P) + a.w_reg*reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0)
        opt.step()
        if it%50==0 or it==a.steps-1:
            rms_line=np.degrees(np.sqrt((ll/npts).item()))
            rms_ov=np.degrees(np.sqrt((ov/len(P)).item()))
            hist.append((it,loss.item(),rms_line,rms_ov))
        if it%1000==0:
            print(f"  step {it:5d}  loss={loss.item():.3e}  line={np.degrees(np.sqrt((ll/npts).item())):.3f}deg  overlap={np.degrees(np.sqrt((ov/len(P)).item())):.3f}deg")

    ll,npts,ov,ov_rob,R12=eval_losses()
    rms_line=np.degrees(np.sqrt((ll/npts).item())); rms_ov=np.degrees(np.sqrt((ov/len(P)).item()))

    # baseline (initial) residuals for comparison
    with torch.no_grad():
        L1b,L2b=Lens(cam1_b,a.N),Lens(cam2_b,a.N)
        llb=torch.zeros(()); nb=0
        for cam,lens in [(1,L1b),(2,L2b)]:
            for pts in lines[cam]:
                D=lens.unproj(pts[:,0],pts[:,1]); D=D/torch.linalg.norm(D,dim=1,keepdim=True)
                llb=llb+smallest_plane_eig(D); nb+=len(pts)
        r1=L1b.unproj(P[:,0],P[:,1]); r2=L2b.unproj(P[:,2],P[:,3]); ovb=((r1-r2@R0)**2).sum()
        base_line=np.degrees(np.sqrt((llb/nb).item())); base_ov=np.degrees(np.sqrt((ovb/len(P)).item()))

    # save params
    def dump(lens):
        fx,fy,cx,cy,k=lens.params()
        return dict(fx=fx.item(),fy=fy.item(),cx=cx.item(),cy=cy.item(),k=[x.item() for x in k])
    fit=dict(N=a.N,cam1=dump(L1),cam2=dump(L2),R12=R12.detach().numpy().tolist(),
             rms_line_deg=rms_line,rms_overlap_deg=rms_ov,
             base_line_deg=base_line,base_overlap_deg=base_ov)
    json.dump(fit,open(f"{OUT}/fitted_params.json","w"),indent=2)
    print(f"LINE straightness:  base={base_line:.3f}deg -> fit={rms_line:.3f}deg")
    print(f"OVERLAP agreement:  base={base_ov:.3f}deg -> fit={rms_ov:.3f}deg")

    # convergence plot
    h=np.array(hist)
    fig,ax=plt.subplots(1,2,figsize=(16,5))
    ax[0].semilogy(h[:,0],h[:,1]); ax[0].set_title("total loss"); ax[0].set_xlabel("step")
    ax[1].plot(h[:,0],h[:,2],label="line straightness (deg)"); ax[1].plot(h[:,0],h[:,3],label="overlap (deg)")
    ax[1].axhline(base_line,ls='--',c='C0',alpha=.5); ax[1].axhline(base_ov,ls='--',c='C1',alpha=.5)
    ax[1].set_title("physical residuals"); ax[1].set_xlabel("step"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/convergence.png",dpi=120); plt.close(fig)

    # per-line residual before/after
    def per_line(lensmap):
        out=[]
        for cam in (1,2):
            lens=lensmap[cam]
            for pts in lines[cam]:
                D=lens.unproj(pts[:,0],pts[:,1]); D=D/torch.linalg.norm(D,dim=1,keepdim=True)
                out.append(np.degrees(np.sqrt((smallest_plane_eig(D)/len(pts)).item())))
        return np.array(out)
    with torch.no_grad():
        rb=per_line({1:L1b,2:L2b}); ra=per_line({1:L1,2:L2})
    x=np.arange(len(rb))
    plt.figure(figsize=(13,5))
    plt.bar(x-0.2,rb,0.4,label=f'base (mean {rb.mean():.2f} deg)',color='indianred')
    plt.bar(x+0.2,ra,0.4,label=f'fitted (mean {ra.mean():.2f} deg)',color='steelblue')
    plt.xlabel('line #'); plt.ylabel('straightness residual (deg)'); plt.legend()
    plt.title('Per-line straightness: base vs fitted'); plt.tight_layout()
    plt.savefig(f"{OUT}/line_residuals.png",dpi=120); plt.close()
    print(f"WROTE {OUT}/fitted_params.json, convergence.png, line_residuals.png")


if __name__=="__main__":
    main()
