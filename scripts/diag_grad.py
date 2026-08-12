#!/usr/bin/env python3
"""Gradient diagnostic: is the solver divergence caused by exploding gradients
through the unrolled-Newton unprojection? Prints per-param grad norms over the
first steps, isolates line-vs-overlap grads, and compares unrolled-Newton grad
vs an implicit (detach+1-step) grad on the SAME pixels."""
import os,sys,json,re,numpy as np,torch
sys.path.insert(0,os.path.dirname(__file__))
import fit_lens as FL
torch.set_default_dtype(torch.float64)

ROOT="/home/raghav/workspace/INSV_STITCHING"
ld=json.load(open(f"{ROOT}/outputs/armC/manual_lines/lines_729.json"))
pd=json.load(open(f"{ROOT}/outputs/armC/manual_pairs/pairs_729.json"))
cam1_b,cam2_b=ld["cam1"],ld["cam2"]; N=6
lines={1:[],2:[]}
for L in ld["lines"]: lines[L["cam"]].append(torch.tensor(L["pts"],dtype=torch.float64))
P=torch.tensor([[p["u1"],p["v1"],p["u2"],p["v2"]] for p in pd["pairs"]])
R0=torch.tensor(FL.recover_R12())
L1,L2=FL.Lens(cam1_b,N),FL.Lens(cam2_b,N); drot=torch.nn.Parameter(torch.zeros(3))
params=list(L1.parameters())+list(L2.parameters())+[drot]
names=["L1.sf","L1.dc","L1.k","L2.sf","L2.dc","L2.k","drot"]

def line_loss():
    ll=torch.zeros(());npts=0
    for cam,lens in [(1,L1),(2,L2)]:
        for pts in lines[cam]:
            D=lens.unproj(pts[:,0],pts[:,1]);D=D/torch.linalg.norm(D,dim=1,keepdim=True)
            ll=ll+FL.smallest_plane_eig(D);npts+=len(pts)
    return ll/npts
def ov_loss():
    R12=FL.so3_exp(drot)@R0
    r1=L1.unproj(P[:,0],P[:,1]);r2=L2.unproj(P[:,2],P[:,3])
    return ((r1-r2@R12)**2).sum()/len(P)

print("=== isolate grad norms at step 0 ===")
for tag,fn in [("LINE",line_loss),("OVERLAP",ov_loss)]:
    for p in params:
        if p.grad is not None: p.grad=None
    l=fn(); l.backward()
    gs=[float(p.grad.norm()) if p.grad is not None else 0 for p in params]
    print(f"{tag:8s} loss={float(l):.3e}  grad norms: "+" ".join(f"{n}={g:.2e}" for n,g in zip(names,gs)))

print("\n=== 8 Adam steps (weighted loss like the fit) ===")
opt=torch.optim.Adam(params,lr=4e-4)
for it in range(8):
    opt.zero_grad()
    ll=line_loss(); ov=ov_loss()
    loss=1e6*ll+ov
    loss.backward()
    gtot=torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    nan=any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in params if p.grad is not None)
    print(f"step {it}: loss={float(loss):.3e} line={np.degrees(np.sqrt(float(ll))):.3f}deg |grad|={float(gtot):.3e} NaN/Inf={nan}")
    torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()

print("\n=== unprojection grad: UNROLLED-NEWTON vs IMPLICIT (detach+1step) ===")
# use overlap pixels of cam2, grad wrt k
u=P[:,2].clone(); v=P[:,3].clone()
def implicit_unproj(u,v,fx,fy,cx,cy,k):
    xp=(u-cx)/fx; yp=(v-cy)/fy; thd=torch.sqrt(xp*xp+yp*yp)+1e-12
    with torch.no_grad():
        th=thd.clone()
        for _ in range(30):
            poly=torch.ones_like(th);dpoly=torch.zeros_like(th)
            for j in range(1,len(k)+1): poly=poly+k[j-1]*th**(2*j);dpoly=dpoly+k[j-1]*(2*j)*th**(2*j-1)
            th=th-(th*poly-thd)/(poly+th*dpoly)
    poly=torch.ones_like(th);dpoly=torch.zeros_like(th)
    for j in range(1,len(k)+1): poly=poly+k[j-1]*th**(2*j);dpoly=dpoly+k[j-1]*(2*j)*th**(2*j-1)
    th=th-(th*poly-thd)/(poly+th*dpoly)
    phi=torch.atan2(yp,xp)
    return torch.stack([torch.sin(th)*torch.cos(phi),torch.sin(th)*torch.sin(phi),torch.cos(th)],-1)

for tag,fn in [("unrolled-Newton(18)",FL.kb_unproject),("implicit(detach+1)",implicit_unproj)]:
    k=torch.tensor(list(cam2_b["k"])+[0.,0.],requires_grad=True)
    out=fn(u,v,cam2_b["fx"],cam2_b["fy"],cam2_b["cx"],cam2_b["cy"],k)
    out.sum().backward()
    print(f"{tag:22s}: d(sum ray)/dk norm = {float(k.grad.norm()):.4e}   max|grad|={float(k.grad.abs().max()):.4e}")
