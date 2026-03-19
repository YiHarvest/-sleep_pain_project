import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

TRAIN_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
IMP_CSV   = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_fulltrain\global_importance_raw.csv"
OUT_TABLE = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_fulltrain\TableA_train_top15.csv"

train = pd.read_csv(TRAIN_CSV)
imp   = pd.read_csv(IMP_CSV)
top15 = imp["feature"].head(15).tolist()

def bh_fdr(p):
    p=np.asarray(p,float); n=len(p)
    order=np.argsort(p); ranked=p[order]
    q=ranked*n/(np.arange(1,n+1))
    q=np.minimum.accumulate(q[::-1])[::-1]
    out=np.empty(n); out[order]=np.clip(q,0,1); return out

pain = train[train["Chronic_pain"]==1]
nop  = train[train["Chronic_pain"]!=1]

rows=[]; pvals=[]
for f in top15:
    x1=pain[f].dropna().values
    x0=nop[f].dropna().values
    u,p=mannwhitneyu(x1,x0,alternative="two-sided")
    r=1-2*u/(len(x1)*len(x0))
    med0=np.median(x0); med1=np.median(x1)
    iqr0=np.quantile(x0,[.25,.75]); iqr1=np.quantile(x1,[.25,.75])
    direction="Higher in pain" if med1>med0 else "Lower in pain"
    rows.append([f,f"{med0:.4g}[{iqr0[0]:.4g},{iqr0[1]:.4g}]",
                   f"{med1:.4g}[{iqr1[0]:.4g},{iqr1[1]:.4g}]",
                   direction,p,r])
    pvals.append(p)

res=pd.DataFrame(rows,columns=["Feature","No-pain median[IQR]","Pain median[IQR]","Direction","p","r"])
res["q_FDR"]=bh_fdr(res["p"].values)
res.to_csv(OUT_TABLE,index=False,encoding="utf-8-sig")
