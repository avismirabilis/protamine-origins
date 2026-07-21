#!/usr/bin/env python3
"""Verification of the expanded PGLS: conditioning + drop-one jackknife + independent GLS recompute."""
import numpy as np
from scipy import stats
import phylo_v2_run as P   # imports rebuild everything (functions, trees, maps)

T=352.0
TIPS,internal,C=P.build_VCV(P.tree46,T)
st46,prop=P.load_states("results_v2/properties.csv",P.TIP46)
sp=[P.TIP46[t] for t in TIPS]
y=np.array([float(prop[s]["charge_density"]) for s in sp])
system=np.array([prop[s]["system"] for s in sp])
partial=np.array([prop[s]["partial"]=="True" for s in sp])
is_p=np.array([s=="Pleurodeles waltl" for s in sp])

def fit(keep):
    Cs=C[np.ix_(keep,keep)]; Rs=Cs/Cs.max()
    X=np.column_stack([np.ones(len(keep)),(system[keep]=="electrostatic").astype(float)])
    return P.pgls(y[keep],X,Rs),Rs

print("="*60)
idx=list(np.where(~partial)[0])                       # PRIMARY: complete seqs
res,Rs=fit(idx)
Rl=P.lambda_corr(Rs,res["lambda"])
print("PRIMARY charge~system (complete seqs, n=%d)"%len(idx))
print("  lambda=%.3f beta_electro=%.4f SE=%.4f p=%.4f"%(res["lambda"],res["beta"][1],res["se"][1],res["p"][1]))
print("  cond(Rl)=%.1f  min eig=%.4f  (well-conditioned if eig>0, cond modest)"%(np.linalg.cond(Rl),np.linalg.eigvalsh(Rl).min()))

# independent GLS recompute (closed form) at the ML lambda
Ri=np.linalg.inv(Rl); X=np.column_stack([np.ones(len(idx)),(system[idx]=="electrostatic").astype(float)])
yy=y[idx]; b=np.linalg.solve(X.T@Ri@X,X.T@Ri@yy); resid=yy-X@b; dof=len(idx)-2
s2=float(resid.T@Ri@resid)/dof; cov=s2*np.linalg.inv(X.T@Ri@X); se=np.sqrt(np.diag(cov))
tval=b[1]/se[1]; p=2*stats.t.sf(abs(tval),dof)
print("  independent recompute: beta=%.4f SE=%.4f t=%.3f p=%.4f  (must match above)"%(b[1],se[1],tval,p))

# drop-one jackknife
betas=[];ps=[]
for d in range(len(idx)):
    keep=[k for kk,k in enumerate(idx) if kk!=d]
    r,_=fit(keep); betas.append(r["beta"][1]); ps.append(r["p"][1])
print("\nDROP-ONE JACKKNIFE (primary, %d fits):"%len(idx))
print("  beta range [%.4f, %.4f]  all positive: %s"%(min(betas),max(betas),all(b>0 for b in betas)))
print("  p range [%.4f, %.4f]  n<0.05: %d/%d  n<0.10: %d/%d"%(min(ps),max(ps),sum(p<0.05 for p in ps),len(ps),sum(p<0.10 for p in ps),len(ps)))
# which drops (if any) lose significance
lost=[(sp[idx[d]],round(ps[d],4)) for d in range(len(idx)) if ps[d]>=0.05]
print("  taxa whose removal pushes p>=0.05:",lost if lost else "NONE")

# no-pleuro
idx2=[k for k in idx if not is_p[k]]; r2,_=fit(idx2)
print("\nno-Pleurodeles: beta=%.4f p=%.4f (n=%d)"%(r2["beta"][1],r2["p"][1],len(idx2)))
