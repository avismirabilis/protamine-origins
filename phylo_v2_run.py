#!/usr/bin/env python3
"""
phylo_v2_run.py -- EXPANDED (46-taxon) re-analysis, per pre-registration
docs/science-superpowers/preregistrations/2026-07-18-protamine-disulfide-origin-expansion.md
Reuses ORIGINAL methods (PGLS ML lambda; phylANOVA Garland 1993; Fitch parsimony) and ADDS an
equal-rates 2-state Mk ML ASR (Felsenstein pruning + marginal reconstruction), validated on the
original 25-taxon tree first. Outputs: results_v2/phylo_results.json
"""
import json, os, csv
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))

TIP25={ "human":"Homo sapiens","gorilla":"Gorilla gorilla","mouse":"Mus musculus","rat":"Rattus norvegicus",
 "guineapig":"Cavia porcellus","rabbit":"Oryctolagus cuniculus","pig":"Sus scrofa","cattle":"Bos taurus",
 "horse":"Equus caballus","bat":"Eptesicus fuscus","opossum":"Monodelphis domestica","devil":"Sarcophilus harrisii",
 "kangaroo":"Macropus giganteus","brushtail":"Trichosurus vulpecula","bandicoot":"Isoodon macrourus",
 "platypus":"Ornithorhynchus anatinus","echidna":"Tachyglossus aculeatus","chicken":"Gallus gallus",
 "quail":"Coturnix japonica","turtle":"Chrysemys picta bellii","snake":"Pantherophis quadrivittatus",
 "anole":"Anolis carolinensis","toad":"Bufo japonicus","newt":"Cynops pyrrhogaster","iberiannewt":"Pleurodeles waltl"}
_mars25=(84,["opossum",(75,["bandicoot",(62,["devil",(45,["kangaroo","brushtail"])])])])
_euth25=(96,[(90,[(9,["human","gorilla"]),(82,["rabbit",(70,[(20,["mouse","rat"]),"guineapig"])])]),
             (85,["bat",(80,["horse",(62,["pig","cattle"])])])])
tree25=(352,[(292,["toad",(66,["newt","iberiannewt"])]),
             (318,[(187,[(55,["platypus","echidna"]),(160,[_mars25,_euth25])]),
                   (280,[(175,["anole","snake"]),(255,["turtle",(35,["chicken","quail"])])])])])

TIP46=dict(TIP25)
TIP46.update({
 "pl_gilesi":"Planigale gilesi","pl_tenui":"Planigale tenuirostris","pl_ingrami":"Planigale ingrami",
 "pl_maculata":"Planigale maculata sinualis","antechinomys":"Antechinomys laniger","antechinus":"Antechinus bellus",
 "dasykaluta":"Dasykaluta rosamondae","dasyurus":"Dasyurus albopunctatus","pseudantechinus":"Pseudantechinus bilarni",
 "sminthopsis":"Sminthopsis bindi","caenolestes":"Caenolestes fuliginosus","didelphis":"Didelphis virginiana",
 "notamacropus":"Notamacropus rufogriseus","dromiciops":"Dromiciops gliroides","myrmecobius":"Myrmecobius fasciatus",
 "notoryctes":"Notoryctes typhlops","perameles":"Perameles gunnii","phascolarctos":"Phascolarctos cinereus",
 "potorous":"Potorous longipes","thylacinus":"Thylacinus cynocephalus","alligator":"Alligator mississippiensis"})
planigale     =(8,[(3,["pl_gilesi","pl_tenui"]),(5,["pl_ingrami","pl_maculata"])])
sminthopsini  =(13,["sminthopsis","antechinomys"])
sminthopsinae =(22,[planigale,sminthopsini])
dasyurini     =(16,[(13,["dasyurus","devil"]),(12,["dasykaluta","pseudantechinus"])])
dasyurinae    =(24,[dasyurini,"antechinus"])
dasyuridae    =(27,[sminthopsinae,dasyurinae])
m34           =(34,["myrmecobius",dasyuridae])
dasyuromorphia=(40,["thylacinus",m34])
m54           =(54,[(30,["bandicoot","perameles"]),dasyuromorphia])
m62           =(62,["notoryctes",m54])
macropodoidea =(40,[(12,["kangaroo","notamacropus"]),"potorous"])
diprotodontia =(55,["phascolarctos",(48,["brushtail",macropodoidea])])
eomarsupialia =(68,[diprotodontia,m62])
australidelphia=(72,["dromiciops",eomarsupialia])
marsupialia   =(82,[(30,["didelphis","opossum"]),(78,["caenolestes",australidelphia])])
glires        =(82,["rabbit",(70,[(20,["mouse","rat"]),"guineapig"])])
eutheria      =(96,[(90,[(9,["human","gorilla"]),glires]),(85,["bat",(80,["horse",(62,["pig","cattle"])])])])
mammalia      =(187,[(55,["platypus","echidna"]),(160,[marsupialia,eutheria])])
archelosauria =(255,["turtle",(240,["alligator",(20,["chicken","quail"])])])
sauropsida    =(280,[(175,["anole","snake"]),archelosauria])
tree46        =(352,[(292,["toad",(66,["newt","iberiannewt"])]),(318,[mammalia,sauropsida])])

def collect_internal(tree):
    internal=[]
    def walk(node):
        if isinstance(node,str): return [node]
        age,ch=node; ts=[]
        for c in ch: ts+=walk(c)
        internal.append((age,frozenset(ts))); return ts
    return sorted(walk(tree)), internal
def mrca_age(internal,a,b):
    best=None
    for age,ts in internal:
        if a in ts and b in ts and (best is None or age<best): best=age
    return best
def build_VCV(tree,T):
    TIPS,internal=collect_internal(tree); n=len(TIPS); C=np.zeros((n,n))
    for i,a in enumerate(TIPS):
        for j,b in enumerate(TIPS):
            C[i,j]=T if i==j else (T-mrca_age(internal,a,b))
    return TIPS,internal,C
def lambda_corr(R,lam):
    Rl=lam*R.copy(); np.fill_diagonal(Rl,1.0); return Rl
def pgls(y,X,R):
    def negll(lam):
        Rl=lambda_corr(R,lam)
        try: Ri=np.linalg.inv(Rl)
        except np.linalg.LinAlgError: return 1e9
        b=np.linalg.solve(X.T@Ri@X,X.T@Ri@y); resid=y-X@b; s2=float(resid.T@Ri@resid)/len(y)
        sign,logdet=np.linalg.slogdet(Rl); return 0.5*(len(y)*np.log(2*np.pi*s2)+logdet+len(y))
    opt=minimize_scalar(negll,bounds=(0,1),method="bounded"); lam=float(opt.x)
    Rl=lambda_corr(R,lam); Ri=np.linalg.inv(Rl); XtRiX=X.T@Ri@X; b=np.linalg.solve(XtRiX,X.T@Ri@y)
    resid=y-X@b; dof=len(y)-X.shape[1]; s2=float(resid.T@Ri@resid)/dof
    se=np.sqrt(np.diag(s2*np.linalg.inv(XtRiX))); tval=b/se; pval=2*stats.t.sf(np.abs(tval),dof)
    return {"lambda":round(lam,3),"beta":[round(x,4) for x in b],"se":[round(x,4) for x in se],
            "t":[round(x,3) for x in tval],"p":[float(f"{x:.3g}") for x in pval],"dof":dof}
def aov_F(y,groups):
    gl=np.unique(groups); grand=y.mean()
    ssb=sum((groups==g).sum()*(y[groups==g].mean()-grand)**2 for g in gl)
    ssw=sum(((y[groups==g]-y[groups==g].mean())**2).sum() for g in gl)
    return (ssb/(len(gl)-1))/(ssw/(len(y)-len(gl)))
def phylanova(y,groups,R,nsim=10000,seed=42):
    F_obs=aov_F(y,groups); rng=np.random.default_rng(seed); L=np.linalg.cholesky(R)
    sims=(L@rng.standard_normal((R.shape[0],nsim)))*np.std(y)+y.mean()
    Fnull=np.array([aov_F(sims[:,k],groups) for k in range(nsim)])
    F_std,p_std=stats.f_oneway(*[y[groups==g] for g in np.unique(groups)])
    return {"F_obs":round(F_obs,3),"p_phylo":float((Fnull>=F_obs).mean()),"p_standard":float(f"{p_std:.3g}"),"nsim":nsim}

def parse_nodes(tree):
    nodes=[]
    def add(node,parent):
        idx=len(nodes)
        if isinstance(node,str): nodes.append(dict(age=0.0,parent=parent,children=[],tip=node)); return idx
        age,ch=node; nodes.append(dict(age=float(age),parent=parent,children=[],tip=None))
        for c in ch: nodes[idx]["children"].append(add(c,idx))
        return idx
    add(tree,-1)
    for nd in nodes: nd["bl"]=(nodes[nd["parent"]]["age"]-nd["age"]) if nd["parent"]>=0 else 0.0
    return nodes
def Pmat(q,t): e=np.exp(-2*q*t); return np.array([[0.5*(1+e),0.5*(1-e)],[0.5*(1-e),0.5*(1+e)]])
def mk_up(nodes,states,q):
    up=[None]*len(nodes)
    def rec(i):
        nd=nodes[i]
        if nd["tip"] is not None:
            v=np.zeros(2); v[states[nd["tip"]]]=1.0; up[i]=v; return v
        v=np.ones(2)
        for c in nd["children"]: v=v*(Pmat(q,nodes[c]["bl"])@rec(c))
        up[i]=v; return v
    rec(0); return up
def mk_loglik(nodes,states,q,prior=(0.5,0.5)): return np.log(np.dot(prior,mk_up(nodes,states,q)[0]))
def mk_marginal(nodes,states,q,prior=(0.5,0.5)):
    up=mk_up(nodes,states,q); down=[None]*len(nodes); down[0]=np.array(prior,float)
    def rec(i):
        for c in nodes[i]["children"]:
            A=down[i].copy()
            for s in nodes[i]["children"]:
                if s!=c: A=A*(Pmat(q,nodes[s]["bl"])@up[s])
            down[c]=A@Pmat(q,nodes[c]["bl"]); rec(c)
    rec(0)
    return {i:(up[i]*down[i])/(up[i]*down[i]).sum() for i in range(len(nodes))}
def mk_fit(nodes,states,prior=(0.5,0.5)):
    opt=minimize_scalar(lambda q:-mk_loglik(nodes,states,q,prior),bounds=(1e-6,0.2),method="bounded")
    return float(opt.x),-opt.fun
def fitch(nodes,states):
    sets=[None]*len(nodes); cost=[0]
    def down(i):
        nd=nodes[i]
        if nd["tip"] is not None: sets[i]={states[nd["tip"]]}; return sets[i]
        chs=[down(c) for c in nd["children"]]; inter=set.intersection(*chs)
        sets[i]=inter if inter else set.union(*chs)
        if not inter: cost[0]+=1
        return sets[i]
    down(0)
    assign=[None]*len(nodes); root=0 if 0 in sets[0] else min(sets[0]); assign[0]=root
    def up(i,ps):
        for c in nodes[i]["children"]:
            s=sets[c]; assign[c]=ps if ps in s else (0 if 0 in s else min(s)); up(c,assign[c])
    up(0,root); gains=losses=0; gnodes=[]
    for i,nd in enumerate(nodes):
        if nd["parent"]>=0:
            ps,cs=assign[nd["parent"]],assign[i]
            if ps==0 and cs==1: gains+=1; gnodes.append(i)
            if ps==1 and cs==0: losses+=1
    return cost[0],gains,losses,root,gnodes
def tips_below(nodes,i):
    nd=nodes[i]
    if nd["tip"] is not None: return {nd["tip"]}
    s=set()
    for c in nd["children"]: s|=tips_below(nodes,c)
    return s
def node_for_tips(nodes,tipset):
    ts=set(tipset)
    for i in range(len(nodes)):
        if tips_below(nodes,i)==ts: return i
    return None
def load_states(csvp,tipmap):
    rows={r["species"]:r for r in csv.DictReader(open(csvp))}
    return {sh:(1 if rows[sp]["system"]=="disulfide" else 0) for sh,sp in tipmap.items()}, rows

out={}
# PART A: validate on 25
st25,_=load_states(os.path.join(HERE,"results","properties.csv"),TIP25)
n25=parse_nodes(tree25); q25,_=mk_fit(n25,st25); p25=mk_marginal(n25,st25,q25)
c25,g25,l25,r25,_=fitch(n25,st25)
out["VALIDATION_25taxon"]={"mk_q_per_Myr":float(f"{q25:.3g}"),"mk_root_P_cysteine_free":round(float(p25[0][0]),4),
 "parsimony_gains":g25,"parsimony_losses":l25,"parsimony_min_changes":c25,"parsimony_root_cysteine_free":r25==0,
 "note":"must reproduce manuscript root ~0.99, q~7.3e-4, single eutherian gain"}
# PART B: expanded 46
T=352.0; TIPS,internal,C=build_VCV(tree46,T); n=len(TIPS)
st46,prop=load_states(os.path.join(HERE,"results_v2","properties.csv"),TIP46)
sp=[TIP46[t] for t in TIPS]
y=np.array([float(prop[s]["charge_density"]) for s in sp]); cys=np.array([float(prop[s]["C"]) for s in sp])
clade=np.array([prop[s]["clade"] for s in sp]); system=np.array([prop[s]["system"] for s in sp])
partial=np.array([prop[s]["partial"]=="True" for s in sp]); is_p=np.array([s=="Pleurodeles waltl" for s in sp])
def suite(mask):
    idx=np.where(mask)[0]; Cs=C[np.ix_(idx,idx)]; Rs=Cs/Cs.max()
    yy=y[idx]; sy=system[idx]; cc=cys[idx]; cl=clade[idx]
    X1=np.column_stack([np.ones(len(idx)),(sy=="electrostatic").astype(float)])
    X3=np.column_stack([np.ones(len(idx)),cc])
    return {"n":int(len(idx)),
      "charge_vs_system":{**pgls(yy,X1,Rs),"note":"beta[1]=electrostatic vs disulfide; POSITIVE=>cys-free higher charge (compensation)"},
      "charge_vs_cysteine":{**pgls(yy,X3,Rs),"note":"beta[1]=slope on cys count; NEGATIVE=>trade-off"},
      "phylANOVA_clade":phylanova(yy,cl,Rs),
      "group_means_charge":{g:round(float(yy[cl==g].mean()),3) for g in np.unique(cl)}}
out["PRIMARY_complete_seqs_with_pleurodeles"]=suite(~partial)
out["SENS_complete_seqs_no_pleurodeles"]=suite((~partial)&(~is_p))
out["SENS_all_including_partials"]=suite(np.ones(n,bool))
DAS_PLAN=["Planigale gilesi","Planigale tenuirostris","Planigale ingrami","Planigale maculata sinualis"]
DAS_FREE=["Antechinomys laniger","Antechinus bellus","Dasykaluta rosamondae","Dasyurus albopunctatus",
          "Pseudantechinus bilarni","Sminthopsis bindi","Sarcophilus harrisii"]
pc=[float(prop[s]["charge_density"]) for s in DAS_PLAN]; fc=[float(prop[s]["charge_density"]) for s in DAS_FREE]
u,pmw=stats.mannwhitneyu(pc,fc,alternative="less")
out["within_Dasyuridae_contrast"]={"planigale_mean":round(float(np.mean(pc)),3),"planigale_values":[round(x,3) for x in pc],
 "sister_mean":round(float(np.mean(fc)),3),"sister_values":[round(x,3) for x in fc],
 "difference":round(float(np.mean(fc)-np.mean(pc)),3),"mannwhitney_U":float(u),"p_one_sided_planigale_lower":float(f"{pmw:.3g}"),
 "note":"independent within-family test: is the independently-derived cys-bearing Planigale LOWER charge than cys-free dasyurid relatives?"}
nodes=parse_nodes(tree46); q46,_=mk_fit(nodes,st46); post=mk_marginal(nodes,st46,q46)
KEY={"root_Tetrapoda":list(TIPS),
 "Amniota":[t for t in TIPS if prop[TIP46[t]]["clade"]!="amphibian"],
 "Mammalia":[t for t in TIPS if prop[TIP46[t]]["clade"] in ("eutherian","marsupial","monotreme")],
 "Theria":[t for t in TIPS if prop[TIP46[t]]["clade"] in ("eutherian","marsupial")],
 "Metatheria":[t for t in TIPS if prop[TIP46[t]]["clade"]=="marsupial"],
 "Eutheria":[t for t in TIPS if prop[TIP46[t]]["clade"]=="eutherian"],
 "Planigale_crown":["pl_gilesi","pl_tenui","pl_ingrami","pl_maculata"],
 "Sminthopsinae":["pl_gilesi","pl_tenui","pl_ingrami","pl_maculata","sminthopsis","antechinomys"]}
mkn={}
for lab,ts in KEY.items():
    ni=node_for_tips(nodes,ts)
    mkn[lab]=({"age_Mya":nodes[ni]["age"],"P_cysteine_free":round(float(post[ni][0]),4),"P_disulfide":round(float(post[ni][1]),4)} if ni is not None else "NOT FOUND")
sens={}
for name,pr in [("flat_0.5",(0.5,0.5)),("biased_disulfide_0.9",(0.1,0.9)),("fixed_cysteinefree",(0.999,0.001))]:
    qx,_=mk_fit(nodes,st46,prior=pr); px=mk_marginal(nodes,st46,qx,prior=pr)
    sens[name]={"q":float(f"{qx:.3g}"),"root_P_cysteine_free":round(float(px[0][0]),4)}
ca,ga,la,ra,gn=fitch(nodes,st46)
out["ASR_expanded"]={"mk_q_per_Myr":float(f"{q46:.3g}"),"mk_marginal_key_nodes":mkn,"mk_root_prior_sensitivity":sens,
 "parsimony_gains":ga,"parsimony_losses":la,"parsimony_min_changes":ca,"parsimony_root_cysteine_free":ra==0,
 "independent_gain_clades":[sorted(tips_below(nodes,gi)) for gi in gn],
 "interpretation":"number of independent origins of disulfide capability = parsimony_gains"}
os.makedirs(os.path.join(HERE,"results_v2"),exist_ok=True)
json.dump(out,open(os.path.join(HERE,"results_v2","phylo_results.json"),"w"),indent=2)

print("="*68); print("PART A - Mk validation on ORIGINAL 25-taxon tree")
print(f"  q_ML={q25:.3g}/Myr  root P(cys-free)={p25[0][0]:.4f}  gains={g25} losses={l25} root_cysfree={r25==0}")
print(f"  (manuscript: root 0.99, q 7.3e-4, single eutherian origin)")
print("="*68); print("PART B - EXPANDED 46-taxon analysis")
print(f"[Q1 ASR] independent origins (parsimony gains)={ga}, losses={la}")
for g in out['ASR_expanded']['independent_gain_clades']: print(f"    gain: {g}")
print(f"  Mk root P(cys-free)={mkn['root_Tetrapoda']['P_cysteine_free']}, Metatheria={mkn['Metatheria']['P_cysteine_free']}, "
      f"Planigale P(disulf)={mkn['Planigale_crown']['P_disulfide']}, Eutheria P(disulf)={mkn['Eutheria']['P_disulfide']}")
print(f"  root-prior sensitivity: {sens}")
pr=out["PRIMARY_complete_seqs_with_pleurodeles"]
print(f"[Q2 charge~system] PRIMARY n={pr['n']}: lambda={pr['charge_vs_system']['lambda']} beta_electro={pr['charge_vs_system']['beta'][1]} "
      f"SE={pr['charge_vs_system']['se'][1]} p={pr['charge_vs_system']['p'][1]}")
print(f"  charge~cys beta={pr['charge_vs_cysteine']['beta'][1]} p={pr['charge_vs_cysteine']['p'][1]}  |  "
      f"phylANOVA F={pr['phylANOVA_clade']['F_obs']} p_phylo={pr['phylANOVA_clade']['p_phylo']} p_std={pr['phylANOVA_clade']['p_standard']}")
w=out["within_Dasyuridae_contrast"]
print(f"[Q2 within-Dasyuridae] Planigale={w['planigale_mean']} vs sisters={w['sister_mean']} diff={w['difference']} p(1-sided)={w['p_one_sided_planigale_lower']}")
print(f"[Sens no Pleuro] charge~system beta={out['SENS_complete_seqs_no_pleurodeles']['charge_vs_system']['beta'][1]} p={out['SENS_complete_seqs_no_pleurodeles']['charge_vs_system']['p'][1]}")
print("wrote results_v2/phylo_results.json")
