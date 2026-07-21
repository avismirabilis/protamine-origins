#!/usr/bin/env python3
"""
robustness_v2.py -- honest-refinement analyses demanded by the adversarial red-team.
EXPLORATORY (not in the frozen pre-registration): flags labeled below.
A) cysteine-composition confound: is the net-charge deficit of disulfide taxa just
   cysteines displacing arginines, or is there ADDITIONAL (adaptive) charge elevation?
B) origin-collapsed within-Dasyuridae contrast (Planigale as ONE independent unit).
C) lambda-sensitivity of the primary PGLS (ML-lambda vs Brownian lambda=1 vs lambda=0).
D) ARD Mk + parsimony single-origin (Sankoff) cost -> is 1-gain-plus-losses competitive?
Outputs: results_v2/robustness.json
"""
import sys, io, os, csv, json
import numpy as np
from scipy import stats
from scipy.linalg import expm
HERE=os.path.dirname(os.path.abspath(__file__))
_o=sys.stdout; sys.stdout=io.StringIO()
import phylo_v2_run as P          # reuse tree46, TIP46, build_VCV, pgls, lambda_corr, parse_nodes, mk_*, fitch
sys.stdout=_o

prop={r["species"]:r for r in csv.DictReader(open(os.path.join(HERE,"results_v2","properties.csv")))}
T=352.0
TIPS,internal,C=P.build_VCV(P.tree46,T)
sp=[P.TIP46[t] for t in TIPS]
def col(n): return np.array([float(prop[s][n]) for s in sp])
charge=col("charge_density"); R=col("R"); K=col("K"); Ccnt=col("C"); Ln=col("length"); nc=col("net_charge_74")
system=np.array([prop[s]["system"] for s in sp]); clade=np.array([prop[s]["clade"] for s in sp])
partial=np.array([prop[s]["partial"]=="True" for s in sp])
comp=~partial
RK_d=(R+K)/Ln; RKC_d=(R+K+Ccnt)/Ln
# cys-adjusted net charge: Cys at pH7.4 ~ -0.074, Arg ~ +1.0  => replacing each Cys by Arg adds +1.074
cysadj_d=(nc + 1.074*Ccnt)/Ln

def gls_fixed(y,X,Rc,lam):
    Rl=lam*Rc.copy(); np.fill_diagonal(Rl,1.0); Ri=np.linalg.inv(Rl)
    b=np.linalg.solve(X.T@Ri@X,X.T@Ri@y); resid=y-X@b; dof=len(y)-X.shape[1]
    s2=float(resid.T@Ri@resid)/dof; se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@Ri@X)))
    tv=b/se; pv=2*stats.t.sf(abs(tv),dof);
    ll=-0.5*(len(y)*np.log(2*np.pi*s2)+np.linalg.slogdet(Rl)[1]+len(y))
    return float(b[1]),float(se[1]),float(pv[1]),float(ll)

def pgls_on(y,mask):
    idx=np.where(mask)[0]; Cs=C[np.ix_(idx,idx)]; Rs=Cs/Cs.max()
    X=np.column_stack([np.ones(len(idx)),(system[idx]=="electrostatic").astype(float)])
    return P.pgls(y[idx],X,Rs), idx, Rs, X

out={"_note":"EXPLORATORY robustness analyses (not pre-registered); requested by adversarial red-team"}

# ---- A) composition confound ----
A={}
for name,y in [("net_charge_density",charge),("(R+K)/len",RK_d),("(R+K+C)/len",RKC_d),("cys_adjusted_charge_density",cysadj_d)]:
    dis=float(y[comp&(system=="disulfide")].mean()); ele=float(y[comp&(system=="electrostatic")].mean())
    r,_,_,_=pgls_on(y,comp)
    A[name]={"disulfide_mean":round(dis,3),"electrostatic_mean":round(ele,3),
             "PGLS_beta_electro":r["beta"][1],"PGLS_p":r["p"][1],"lambda":r["lambda"]}
out["A_composition_confound"]=A
out["A_interpretation"]=("If net-charge shows electro>disulfide (compensation direction) BUT (R+K+C)/len shows "
  "disulfide>=electro, the net-charge deficit is cysteines occupying basic positions (mechanical), NOT extra "
  "arginine in cysteine-free lineages -> adaptive electrostatic compensation NOT separable from composition.")

# within-Dasyuridae, adjusted metrics
DAS_PLAN=["Planigale gilesi","Planigale tenuirostris","Planigale ingrami","Planigale maculata sinualis"]
DAS_FREE=["Antechinomys laniger","Antechinus bellus","Dasykaluta rosamondae","Dasyurus albopunctatus",
          "Pseudantechinus bilarni","Sminthopsis bindi","Sarcophilus harrisii"]
def m(names,y):
    idxs=[sp.index(n) for n in names]; return float(np.mean([y[i] for i in idxs]))
out["A_within_Dasyuridae"]={
  "net_charge":{"planigale":round(m(DAS_PLAN,charge),3),"sisters":round(m(DAS_FREE,charge),3)},
  "(R+K+C)/len":{"planigale":round(m(DAS_PLAN,RKC_d),3),"sisters":round(m(DAS_FREE,RKC_d),3)},
  "cys_adjusted_charge":{"planigale":round(m(DAS_PLAN,cysadj_d),3),"sisters":round(m(DAS_FREE,cysadj_d),3)}}

# ---- B) origin-collapsed within-Dasyuridae ----
plan_vals=[charge[sp.index(n)] for n in DAS_PLAN]; free_vals=[charge[sp.index(n)] for n in DAS_FREE]
# Planigale as ONE unit vs 7 sisters: prob a single unit is the lowest of 8 = 1/8
out["B_origin_collapsed_contrast"]={
  "planigale_clade_mean":round(float(np.mean(plan_vals)),3),"n_independent_origins_on_derived_side":1,
  "cysteine_free_sister_values":[round(v,3) for v in sorted(free_vals)],
  "all_planigale_below_all_sisters":bool(max(plan_vals)<min(free_vals)),
  "naive_pseudoreplicated_MWU_p":0.003,
  "honest_p_planigale_as_one_unit":"n=1 vs 7 -> no valid test; direction only (Planigale clade below all 7 sisters); combinatorial floor ~0.125",
  "note":"the p=0.003 Mann-Whitney counts 4 congeners (one origin) as 4 points = pseudoreplication; report DIRECTION only"}

# ---- C) lambda sensitivity of primary ----
r,idx,Rs,X=pgls_on(charge,comp); yy=charge[idx]
ml_lam=r["lambda"]
lamC={}
for tag,lam in [("ML_lambda",ml_lam),("Brownian_lambda_1",1.0),("lambda_0_OLSlike",0.0)]:
    b,se,pv,ll=gls_fixed(yy,X,Rs,lam)
    lamC[tag]={"lambda":round(lam,3),"beta_electro":round(b,4),"se":round(se,4),"p":round(pv,4)}
# LRT for the system term at ML lambda
_,_,_,ll_full=gls_fixed(yy,X,Rs,ml_lam)
X0=np.ones((len(idx),1));
def gls_ll_intercept(y,Rc,lam):
    Rl=lam*Rc.copy(); np.fill_diagonal(Rl,1.0); Ri=np.linalg.inv(Rl)
    b=np.linalg.solve(X0.T@Ri@X0,X0.T@Ri@y); resid=y-X0@b; s2=float(resid.T@Ri@resid)/len(y)
    return -0.5*(len(y)*np.log(2*np.pi*s2)+np.linalg.slogdet(Rl)[1]+len(y))
ll_null=gls_ll_intercept(yy,Rs,ml_lam)
lrt=2*(ll_full-ll_null); p_lrt=stats.chi2.sf(lrt,1)
lamC["LRT_system_term"]={"chi2":round(lrt,3),"p":round(float(p_lrt),4)}
out["C_lambda_sensitivity"]=lamC

# ---- D) ARD Mk + Sankoff single-origin cost ----
st46,_=P.load_states(os.path.join(HERE,"results_v2","properties.csv"),P.TIP46)
nodes=P.parse_nodes(P.tree46)
def mk_up_gen(nodes,states,Pfun):
    up=[None]*len(nodes)
    def rec(i):
        nd=nodes[i]
        if nd["tip"] is not None:
            v=np.zeros(2); v[states[nd["tip"]]]=1.0; up[i]=v; return v
        v=np.ones(2)
        for c in nd["children"]: v=v*(Pfun(nodes[c]["bl"])@rec(c))
        up[i]=v; return v
    rec(0); return up
def ard_negll(params):
    q01,q10=np.exp(params)
    Pf=lambda t: expm(np.array([[-q01,q01],[q10,-q10]])*t)
    up=mk_up_gen(nodes,st46,Pf); return -np.log(0.5*up[0][0]+0.5*up[0][1])
from scipy.optimize import minimize
res=minimize(ard_negll,np.log([1e-3,1e-3]),method="Nelder-Mead")
q01,q10=np.exp(res.x); ll_ard=-res.fun
# ER loglik at its ML q
qER,_=P.mk_fit(nodes,st46);
Pf_er=lambda t: expm(np.array([[-qER,qER],[qER,-qER]])*t)
up_er=mk_up_gen(nodes,st46,Pf_er); ll_er=np.log(0.5*up_er[0][0]+0.5*up_er[0][1])
# ARD root marginal (flat prior)
Pf_ard=lambda t: expm(np.array([[-q01,q01],[q10,-q10]])*t)
up_ard=mk_up_gen(nodes,st46,Pf_ard); root_cf_ard=up_ard[0][0]/(up_ard[0][0]+up_ard[0][1])
aic_er=2*1-2*ll_er; aic_ard=2*2-2*ll_ard
# Sankoff single-origin: parsimony min-changes if ROOT forced disulfide (=1). Compare to free (=3)
def fitch_forced_root(nodes,states,forced):
    sets=[None]*len(nodes); cost=[0]
    def down(i):
        nd=nodes[i]
        if nd["tip"] is not None: sets[i]={states[nd["tip"]]}; return sets[i]
        chs=[down(c) for c in nd["children"]]; inter=set.intersection(*chs)
        sets[i]=inter if inter else set.union(*chs)
        if not inter: cost[0]+=1
        return sets[i]
    down(0)
    # if forced root state not achievable at min cost, add 1
    extra=0 if forced in sets[0] else 1
    return cost[0]+extra
free_cost,gains,losses,_,_=P.fitch(nodes,st46)
forced_single=fitch_forced_root(nodes,st46,1)  # root disulfide -> single-origin-ish scenario
out["D_ASR_model_sensitivity"]={
  "ER":{"q":float(f"{qER:.3g}"),"logL":round(float(ll_er),3),"AIC":round(float(aic_er),2),"root_P_cys_free":round(float(P.mk_marginal(nodes,st46,qER)[0][0]),4)},
  "ARD":{"q01_gain":float(f"{q01:.3g}"),"q10_loss":float(f"{q10:.3g}"),"loss_over_gain":float(f"{q10/q01:.2g}"),
         "logL":round(float(ll_ard),3),"AIC":round(float(aic_ard),2),"root_P_cys_free":round(float(root_cf_ard),4)},
  "LRT_ARD_vs_ER":{"2dLL":round(float(2*(ll_ard-ll_er)),3),"AIC_favors":"ER" if aic_er<aic_ard else "ARD","dAIC":round(float(abs(aic_er-aic_ard)),2)},
  "parsimony_free_min_changes":free_cost,"parsimony_gains":gains,"parsimony_losses":losses,
  "parsimony_forced_disulfide_root":forced_single,
  "note":"ER favored by AIC; ARD drives loss~0 and keeps cys-free root; single-origin (disulfide root) requires more changes -> multiple-origins is model-robust"}

json.dump(out,open(os.path.join(HERE,"results_v2","robustness.json"),"w"),indent=2)

# console
print("="*70); print("A) COMPOSITION CONFOUND (complete seqs, disulfide vs electrostatic)")
for k,v in A.items():
    print(f"  {k:26s} disulf={v['disulfide_mean']:.3f} electro={v['electrostatic_mean']:.3f}  PGLS beta={v['PGLS_beta_electro']:+.4f} p={v['PGLS_p']}")
wd=out["A_within_Dasyuridae"]
print(f"  within-Dasyuridae  net_charge: Planigale {wd['net_charge']['planigale']} vs sisters {wd['net_charge']['sisters']}")
print(f"                     (R+K+C)/len: Planigale {wd['(R+K+C)/len']['planigale']} vs sisters {wd['(R+K+C)/len']['sisters']}")
print(f"                     cys-adj chg: Planigale {wd['cys_adjusted_charge']['planigale']} vs sisters {wd['cys_adjusted_charge']['sisters']}")
print("\nC) LAMBDA SENSITIVITY of primary PGLS charge~system:")
for k,v in out["C_lambda_sensitivity"].items():
    if k=="LRT_system_term": print(f"  LRT system term: chi2={v['chi2']} p={v['p']}")
    else: print(f"  {k:20s} lambda={v['lambda']} beta={v['beta_electro']:+.4f} p={v['p']}")
print("\nD) ASR MODEL SENSITIVITY:")
d=out["D_ASR_model_sensitivity"]
print(f"  ER : root P(cys-free)={d['ER']['root_P_cys_free']} AIC={d['ER']['AIC']}")
print(f"  ARD: root P(cys-free)={d['ARD']['root_P_cys_free']} loss/gain={d['ARD']['loss_over_gain']} AIC={d['ARD']['AIC']} (dAIC {d['LRT_ARD_vs_ER']['dAIC']}, favors {d['LRT_ARD_vs_ER']['AIC_favors']})")
print(f"  parsimony: free={d['parsimony_free_min_changes']} ({d['parsimony_gains']} gains/{d['parsimony_losses']} losses) vs forced-disulfide-root={d['parsimony_forced_disulfide_root']}")
print("\nwrote results_v2/robustness.json")
