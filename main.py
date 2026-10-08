import os, time, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CONFIG = {
    "H": 32, "W": 32, "D": 16,
    "n_control": 20, "n_abnormal": 20,
    "effect": 0.8, "noise": 1.0,
    "alpha": 0.05, "k_neighbors": 12,
    "mf_iters": 8,
    "mc_reps": 10,
    "strength_reps": 5,
    "size_reps": 5,
    "noise_reps": 5,
    "roi_epochs": 20,
    "roi_pos_weight": 8.0,
    "seed": 42,
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make the small simulation deterministic where supported.
    torch.use_deterministic_algorithms(False)

def make_brain_mask(H,W,D):
    yy,xx,zz=np.mgrid[:H,:W,:D]
    cy,cx,cz=(H-1)/2,(W-1)/2,(D-1)/2
    val=((yy-cy)/(H*0.45))**2+((xx-cx)/(W*0.45))**2+((zz-cz)/(D*0.48))**2
    return val <= 1.0

def sphere_mask(H,W,D,center,radius,brain):
    yy,xx,zz=np.mgrid[:H,:W,:D]
    cy,cx,cz=center
    m=(yy-cy)**2+(xx-cx)**2+(zz-cz)**2 <= radius**2
    return m & brain

def generate_dataset(effect=0.8, noise=1.0, seed=42, size_scale=1.0,
                     n_control=20, n_abnormal=20):
    seed_all(seed)
    H,W,D=CONFIG["H"],CONFIG["W"],CONFIG["D"]
    brain=make_brain_mask(H,W,D)
    truth=np.zeros((H,W,D),dtype=np.uint8)
    regions=[
        ((10,10,5), max(2,int(round(3.5*size_scale)))),
        ((22,22,9), max(2,int(round(4.2*size_scale)))),
        ((16,12,11), max(2,int(round(3.0*size_scale)))),
    ]
    for c,r in regions: truth |= sphere_mask(H,W,D,c,r,brain).astype(np.uint8)

    n1,n2=n_control,n_abnormal
    control=np.random.normal(0,noise,(n1,H,W,D)).astype(np.float32)
    abnormal=np.random.normal(0,noise,(n2,H,W,D)).astype(np.float32)

    # Mild anatomical/background variation inside the brain.
    baseline=np.random.normal(0,0.08,(H,W,D)).astype(np.float32)*brain
    control += baseline
    abnormal += baseline
    abnormal += effect*truth[None,...]
    return control, abnormal, brain.astype(np.uint8), truth

def save_dataset(path, control, abnormal, brain, truth, meta):
    np.savez_compressed(path, control=control, abnormal=abnormal,
                        brain_mask=brain, truth=truth)
    with open(str(path).replace(".npz",".json"),"w") as f:
        json.dump(meta,f,indent=2)

def voxel_statistics(control, abnormal, brain):
    x1=control[:,brain.astype(bool)]
    x2=abnormal[:,brain.astype(bool)]
    n1,n2=x1.shape[0],x2.shape[0]
    m1,m2=x1.mean(0),x2.mean(0)
    v1=x1.var(0,ddof=1); v2=x2.var(0,ddof=1)
    se=np.sqrt(v1/n1+v2/n2)+1e-8
    z=(m2-m1)/se
    p=2*stats.norm.sf(np.abs(z))
    shape=brain.shape
    zmap=np.zeros(np.prod(shape)); pmap=np.ones(np.prod(shape))
    idx=np.flatnonzero(brain.ravel())
    zmap[idx]=z; pmap[idx]=p
    return zmap.reshape(shape),pmap.reshape(shape)

class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv3d(1,8,3,padding=1), nn.ReLU(),
            nn.Conv3d(8,16,3,padding=1), nn.ReLU(),
            nn.Conv3d(16,16,3,padding=1), nn.ReLU(),
            nn.Conv3d(16,8,1), nn.ReLU(),
            nn.Conv3d(8,1,1)
        )
    def forward(self,x): return self.net(x)

def train_roi_detector(train_effects, train_truths, test_effect, device,
                       epochs=20, seed=1234, pos_weight=8.0):
    """
    Train a small 3D CNN to produce a soft ROI probability map.
    The positive class is weighted because simulated abnormal voxels are
    much fewer than background voxels.
    """
    seed_all(seed)

    X=np.stack(train_effects).astype(np.float32)
    Y=np.stack(train_truths).astype(np.float32)
    Xt=np.stack([test_effect]).astype(np.float32)

    mean,std=X.mean(),X.std()+1e-6
    X=(X-mean)/std
    Xt=(Xt-mean)/std

    model=Tiny3DCNN().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)

    pw=torch.tensor([float(pos_weight)],dtype=torch.float32,device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pw)

    ds=TensorDataset(torch.from_numpy(X[:,None]),
                     torch.from_numpy(Y[:,None]))
    # Fixed order removes avoidable stochasticity between replications.
    dl=DataLoader(ds,batch_size=2,shuffle=False)

    model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        inp=torch.from_numpy(Xt[:,None]).to(device)
        prob=torch.sigmoid(model(inp))[0,0].cpu().numpy()

    # Keep ROI as a soft prior; do not hard-threshold it.
    prob=np.clip(prob,0.02,0.98)
    return model, prob

def bh(p,alpha=0.05):
    p=np.asarray(p); m=len(p)
    order=np.argsort(p); sp=p[order]
    q=alpha*np.arange(1,m+1)/m
    valid=np.where(sp<=q)[0]
    out=np.zeros(m,dtype=bool)
    if len(valid): out[order[:valid[-1]+1]]=True
    return out

def storey_qvalue(p,alpha=0.05):
    p=np.asarray(p); m=len(p)
    lambdas=np.arange(0.05,0.96,0.05)
    pi0=np.min([(np.mean(p>l))/(1-l) for l in lambdas])
    pi0=float(np.clip(pi0,0.05,1.0))
    order=np.argsort(p); sp=p[order]
    q=pi0*m*sp/(np.arange(1,m+1))
    q=np.minimum.accumulate(q[::-1])[::-1]
    out=np.zeros(m,dtype=bool); out[order[q<=alpha]]=True
    return out

def local_fdr(p,alpha=0.05):
    # Two-component Gaussian mixture on z; local null posterior.
    z=stats.norm.isf(np.clip(p/2,1e-12,1-1e-12))*np.sign(0.5-p)
    X=np.abs(z).reshape(-1,1)
    try:
        gm=GaussianMixture(n_components=2,random_state=0).fit(X)
        comp=gm.predict_proba(X)
        means=gm.means_.ravel()
        null_idx=int(np.argmin(means))
        post_null=comp[:,null_idx]
        # Calibrate to be conservative at alpha.
        out=post_null<=alpha
    except Exception:
        out=bh(p,alpha)
    return out

def adaptive_sparse_graph(coords, zvals, k=12):
    """Construct a row-normalized sparse adaptive kNN graph."""
    k=min(k,len(coords)-1)
    nnm=NearestNeighbors(n_neighbors=k+1).fit(coords)
    dist,inds=nnm.kneighbors(coords)
    dist=dist[:,1:]
    inds=inds[:,1:]

    zd=np.abs(zvals)
    sigma_s=max(float(np.median(dist)),1e-3)
    sigma_a=max(float(np.median(np.abs(zd-np.median(zd)))),1e-3)

    rows=np.repeat(np.arange(len(coords)),k)
    cols=inds.reshape(-1)

    spatial=np.exp(-(dist**2)/(2*sigma_s**2))
    appearance=np.exp(-((zd[inds]-zd[:,None])**2)/(2*sigma_a**2))
    weights=(spatial*appearance).reshape(-1)

    row_sum=np.bincount(rows,weights=weights,minlength=len(coords))
    weights=weights/(row_sum[rows]+1e-12)

    W=sparse.csr_matrix((weights,(rows,cols)),
                        shape=(len(coords),len(coords)),
                        dtype=np.float32)
    return W,sigma_s,sigma_a

def fc_hmrf_lis(zvals, coords, alpha=0.05, k=12, iters=8, roi_prob=None):
    """
    Adaptive sparse fcHMRF-LIS approximation.

    The unary term is driven by standardized voxel evidence and the CNN
    ROI probability is used as a bounded soft prior. Mean-field updates
    encourage spatially coherent discoveries without allowing the graph
    term to overwhelm strong voxel-wise evidence.
    """
    W,sig_s,sig_a=adaptive_sparse_graph(coords,zvals,k)

    az=np.abs(zvals)

    # Robust evidence scale.  A |z| around 2 is treated as the transition
    # from weak/background evidence to plausible signal.
    base=1.0/(1.0+np.exp(-2.0*(az-2.0)))
    base=0.01+0.97*base

    if roi_prob is not None:
        rp=np.clip(np.asarray(roi_prob),0.02,0.98)
        # CNN contributes information, but cannot erase statistical evidence.
        qsig=0.70*base + 0.30*rp
    else:
        qsig=base

    qsig=np.clip(qsig,1e-4,1-1e-4)

    # Moderate spatial coupling; bounded so FDR remains driven by evidence.
    smooth=float(np.clip(0.10+0.04*np.std(az),0.10,0.18))

    for _ in range(iters):
        neigh=np.asarray(W.dot(qsig)).ravel()
        logit=np.log(qsig)-np.log1p(-qsig)
        logit += smooth*(2.0*neigh-1.0)
        qsig=1.0/(1.0+np.exp(-np.clip(logit,-20,20)))
        qsig=np.clip(qsig,1e-5,1-1e-5)

    lis=1.0-qsig

    # Standard LIS-FDR step-up rule.
    order=np.argsort(lis)
    c=np.cumsum(lis[order])/(np.arange(len(lis))+1)
    valid=np.where(c<=alpha)[0]

    reject=np.zeros(len(lis),dtype=bool)
    if len(valid):
        reject[order[:valid[-1]+1]]=True

    return reject,lis,{
        "sigma_spatial":sig_s,
        "sigma_appearance":sig_a,
        "smoothness":smooth
    }

def metrics(pred,truth,runtime):
    pred=np.asarray(pred,dtype=bool); truth=np.asarray(truth,dtype=bool)
    tp=int(np.sum(pred&truth)); fp=int(np.sum(pred&~truth))
    fn=int(np.sum(~pred&truth)); tn=int(np.sum(~pred&~truth))
    fdp=fp/max(tp+fp,1); fnr=fn/max(tp+fn,1)
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    dice=2*tp/max(2*tp+fp+fn,1)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,FDP=fdp,FNR=fnr,
                Precision=prec,Recall=rec,Dice= dice,
                Discoveries=int(pred.sum()),Runtime_s=runtime)

def run_methods(control,abnormal,brain,truth,roi_prob=None):
    zmap,pmap=voxel_statistics(control,abnormal,brain)
    mask=brain.astype(bool)
    z=zmap[mask]; p=pmap[mask]; y=truth[mask].astype(bool)
    coords=np.argwhere(mask)
    rows=[]
    for name,func in [
        ("BH",lambda:bh(p,CONFIG["alpha"])),
        ("q-value",lambda:storey_qvalue(p,CONFIG["alpha"])),
        ("LocalFDR",lambda:local_fdr(p,CONFIG["alpha"])),
    ]:
        t=time.perf_counter(); pred=func(); rt=time.perf_counter()-t
        rows.append({"Method":name,**metrics(pred,y,rt)})
    t=time.perf_counter()
    rp=None if roi_prob is None else roi_prob[mask]
    pred,lis,info=fc_hmrf_lis(z,coords,CONFIG["alpha"],CONFIG["k_neighbors"],
                              CONFIG["mf_iters"],rp)
    rt=time.perf_counter()-t
    rows.append({"Method":"Adaptive Sparse fcHMRF-LIS",**metrics(pred,y,rt)})
    predmaps={}
    for name,func in [
        ("BH",lambda:bh(p,CONFIG["alpha"])),
        ("q-value",lambda:storey_qvalue(p,CONFIG["alpha"])),
        ("LocalFDR",lambda:local_fdr(p,CONFIG["alpha"])),
    ]:
        predmaps[name]=func()
    predmaps["Adaptive Sparse fcHMRF-LIS"]=pred
    full={}
    for name,v in predmaps.items():
        a=np.zeros(brain.shape,dtype=np.uint8); a[mask]=v.astype(np.uint8); full[name]=a
    lm=np.zeros(brain.shape); lm[mask]=lis
    return pd.DataFrame(rows),zmap,pmap,lm,full,info

def generate_roi_training(n=12,seed=100):
    effects=[]; truths=[]
    for i in range(n):
        c,a,b,t=generate_dataset(effect=np.random.default_rng(seed+i).uniform(.5,1.2),
                                  noise=1.0,seed=seed+i,size_scale=np.random.default_rng(seed+i).uniform(.8,1.2),
                                  n_control=20,n_abnormal=20)
        z,_=voxel_statistics(c,a,b)
        effects.append(z); truths.append(t)
    return effects,truths

def plot_maps(zmap,roi_prob,lis,truth,full,mid,save):
    fig,axs=plt.subplots(2,4,figsize=(15,7))
    imgs=[truth,zmap,roi_prob,lis,full["BH"],full["q-value"],full["LocalFDR"],full["Adaptive Sparse fcHMRF-LIS"]]
    titles=["Ground Truth","Z-statistic","DL ROI Probability","LIS","BH","q-value","LocalFDR","Adaptive fcHMRF-LIS"]
    for ax,img,title in zip(axs.ravel(),imgs,titles):
        im=ax.imshow(img[:,:,mid]); ax.set_title(title); ax.axis("off"); plt.colorbar(im,ax=ax,fraction=.046)
    plt.tight_layout(); plt.savefig(save,dpi=180); plt.close()

def plot_comparison(df,save):
    summ=df.groupby("Method").agg({c:"mean" for c in ["FDP","FNR","Precision","Recall","Dice","Runtime_s","TP"]}).reset_index()
    x=np.arange(len(summ)); w=.12
    fig,ax=plt.subplots(figsize=(13,6))
    for i,c in enumerate(["FDP","FNR","Precision","Recall","Dice"]):
        ax.bar(x+(i-2)*w,summ[c],w,label=c)
    ax.set_xticks(x); ax.set_xticklabels(summ.Method,rotation=20,ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Mean value"); ax.set_title("Method Comparison"); ax.legend()
    plt.tight_layout(); plt.savefig(save,dpi=180); plt.close()
    summ.to_csv(RESULTS/"main_summary.csv",index=False)

def interactive_3d(truth,pred,zmap,save):
    pts=np.argwhere(pred.astype(bool))
    gt=np.argwhere(truth.astype(bool))
    fig=go.Figure()
    if len(gt):
        fig.add_trace(go.Scatter3d(x=gt[:,1],y=gt[:,0],z=gt[:,2],mode="markers",
                                   marker=dict(size=3),name="Ground Truth"))
    if len(pts):
        fig.add_trace(go.Scatter3d(x=pts[:,1],y=pts[:,0],z=pts[:,2],mode="markers",
                                   marker=dict(size=4),name="Detected"))
    fig.update_layout(title="3D Ground Truth vs Adaptive fcHMRF-LIS Detection",
                      scene=dict(xaxis_title="X",yaxis_title="Y",zaxis_title="Z"))
    fig.write_html(save)

def run_all():
    seed_all(CONFIG["seed"])
    print("\n=== Adaptive fcHMRF-LIS Final Implementation ===")
    print("Simulation-based evaluation; no clinical-data claim.\n")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",device)

    # Independent ROI training simulations and one test dataset.
    tr_eff,tr_truth=generate_roi_training(12,200)
    control,abnormal,brain,truth=generate_dataset(CONFIG["effect"],CONFIG["noise"],CONFIG["seed"])
    zmap,pmap=voxel_statistics(control,abnormal,brain)
    _,roi_prob=train_roi_detector(tr_eff,tr_truth,zmap,device,CONFIG["roi_epochs"])
    np.savez_compressed(DATA/"main_dataset.npz",control=control,abnormal=abnormal,
                        brain_mask=brain,truth=truth,zmap=zmap,pmap=pmap,roi_probability=roi_prob)
    print("Dataset saved:",DATA/"main_dataset.npz")

    main,zm,pm,lis,full,info=run_methods(control,abnormal,brain,truth,roi_prob)
    main.insert(0,"Replication",0)
    main.to_csv(RESULTS/"main_results.csv",index=False)
    plot_maps(zm,roi_prob,lis,truth,full,CONFIG["D"]//2,RESULTS/"discovery_maps.png")
    plot_comparison(main,RESULTS/"method_comparison.png")
    interactive_3d(truth,full["Adaptive Sparse fcHMRF-LIS"],zm,RESULTS/"interactive_3d.html")

    # Monte Carlo main experiment.
    allrows=[]
    for r in range(CONFIG["mc_reps"]):
        c,a,b,t=generate_dataset(CONFIG["effect"],CONFIG["noise"],CONFIG["seed"]+10+r)
        z,_=voxel_statistics(c,a,b)
        _,rp=train_roi_detector(tr_eff,tr_truth,z,device,CONFIG["roi_epochs"],seed=2000+r)
        d,_,_,_,_,_=run_methods(c,a,b,t,rp)
        d.insert(0,"Replication",r)
        allrows.append(d)
    mc=pd.concat(allrows,ignore_index=True)
    mc.to_csv(RESULTS/"monte_carlo_results.csv",index=False)
    mc.groupby("Method").agg({c:["mean","std"] for c in ["FDP","FNR","Precision","Recall","Dice","Runtime_s","TP"]}).to_csv(RESULTS/"monte_carlo_summary.csv")

    # Effect strength experiment.
    rows=[]
    for eff in [.4,.6,.8,1.0,1.2]:
        for r in range(CONFIG["strength_reps"]):
            c,a,b,t=generate_dataset(eff,1.0,5000+int(eff*100)+r)
            z,_=voxel_statistics(c,a,b); _,rp=train_roi_detector(tr_eff,tr_truth,z,device,CONFIG["roi_epochs"],seed=3000+int(eff*100)+r)
            d,*_=run_methods(c,a,b,t,rp); d.insert(0,"Effect",eff); d.insert(1,"Replication",r); rows.append(d)
    pd.concat(rows,ignore_index=True).to_csv(RESULTS/"effect_strength_results.csv",index=False)

    # Size experiment.
    rows=[]
    for scale in [.7,1.0,1.3]:
        for r in range(CONFIG["size_reps"]):
            c,a,b,t=generate_dataset(.8,1.0,7000+r,size_scale=scale)
            z,_=voxel_statistics(c,a,b); _,rp=train_roi_detector(tr_eff,tr_truth,z,device,CONFIG["roi_epochs"],seed=4000+r)
            d,*_=run_methods(c,a,b,t,rp); d.insert(0,"SizeScale",scale); d.insert(1,"Replication",r); rows.append(d)
    pd.concat(rows,ignore_index=True).to_csv(RESULTS/"size_results.csv",index=False)

    # Noise robustness.
    rows=[]
    for noise in [.8,1.0,1.2,1.5]:
        for r in range(CONFIG["noise_reps"]):
            c,a,b,t=generate_dataset(.8,noise,9000+int(noise*100)+r)
            z,_=voxel_statistics(c,a,b); _,rp=train_roi_detector(tr_eff,tr_truth,z,device,CONFIG["roi_epochs"],seed=5000+int(noise*100)+r)
            d,*_=run_methods(c,a,b,t,rp); d.insert(0,"Noise",noise); d.insert(1,"Replication",r); rows.append(d)
    nr=pd.concat(rows,ignore_index=True)
    nr.to_csv(RESULTS/"noise_robustness.csv",index=False)

    # Noise plot.
    g=nr.groupby(["Noise","Method"])["FDP"].mean().reset_index()
    fig,ax=plt.subplots(figsize=(9,5))
    for method,sub in g.groupby("Method"):
        ax.plot(sub.Noise,sub.FDP,marker="o",label=method)
    ax.axhline(CONFIG["alpha"],linestyle="--",label="Target alpha=0.05")
    ax.set_xlabel("Noise standard deviation"); ax.set_ylabel("Mean FDP")
    ax.set_title("Noise Robustness: False Discovery Proportion")
    ax.legend(); plt.tight_layout(); plt.savefig(RESULTS/"noise_robustness.png",dpi=180); plt.close()

    with open(RESULTS/"configuration.json","w") as f: json.dump(CONFIG,f,indent=2)
    print("\nDONE.")
    print("Results folder:",RESULTS)
    print("Open:",RESULTS/"interactive_3d.html")
    print("\nMain results:")
    print(main.to_string(index=False))

if __name__=="__main__":
    run_all()
