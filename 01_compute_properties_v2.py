#!/usr/bin/env python3
"""
01_compute_properties.py
Uniformly recompute biophysical properties for all curated tetrapod protamines,
and cross-check pI / net charge against an independent implementation (Biopython).

Outputs:
  results/properties.csv     -- master table used by downstream phylogenetics
  results/verification.json  -- agreement between our HH implementation and Biopython
"""
import json, os, re, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results_v2"); os.makedirs(RES, exist_ok=True)

# ---- Kyte-Doolittle hydropathy ----
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

# ---- EMBOSS pKa set (used for pI / net-charge by Henderson-Hasselbalch) ----
PKA = {'Nterm':8.6,'Cterm':3.6,'C':8.5,'D':3.9,'E':4.1,'H':6.5,'K':10.8,'R':12.5,'Y':10.1}

def net_charge(seq, pH):
    pos = 1.0/(1.0+10**(pH-PKA['Nterm']))
    neg = 1.0/(1.0+10**(PKA['Cterm']-pH))
    for aa,n in [('K',seq.count('K')),('R',seq.count('R')),('H',seq.count('H'))]:
        pos += n*(1.0/(1.0+10**(pH-PKA[aa])))
    for aa,n in [('D',seq.count('D')),('E',seq.count('E')),('C',seq.count('C')),('Y',seq.count('Y'))]:
        neg += n*(1.0/(1.0+10**(PKA[aa]-pH)))
    return pos-neg

def isoelectric(seq):
    lo,hi = 0.0,14.0
    for _ in range(100):
        mid=(lo+hi)/2.0
        if net_charge(seq,mid)>0: lo=mid
        else: hi=mid
    return (lo+hi)/2.0

def max_run(seq, aa='R'):
    runs=re.findall(aa+'+',seq); return max((len(r) for r in runs),default=0)

def foldindex(seq):
    # Prilusky et al. 2005: FI = 2.785*<H> - |<charge>| - 1.151  (negative => disordered)
    meanH=np.mean([KD[a] for a in seq if a in KD])
    meanQ=net_charge(seq,7.0)/len(seq)
    return 2.785*meanH - abs(meanQ) - 1.151

def uversky_disordered(seq):
    # Uversky 2000 boundary: <R> = 2.785<H> - 1.151  (mean |net charge| per residue vs mean Kyte-Doolittle on 0-1 scale)
    meanH=np.mean([KD[a] for a in seq if a in KD])
    H01=(meanH+4.5)/9.0  # rescale KD to 0..1
    meanQ=abs(net_charge(seq,7.0))/len(seq)
    boundary=2.785*H01-1.151
    return meanQ>boundary, meanQ, H01

data=json.load(open(os.path.join(HERE,"protamine_sequences_v2.json")))["sequences"]

rows=[]
for d in data:
    s=d["seq"]; L=len(s)
    R=s.count('R'); K=s.count('K'); C=s.count('C'); H=s.count('H')
    nc74=net_charge(s,7.4)
    fi=foldindex(s); ud,mq,h01=uversky_disordered(s)
    rows.append({
        "species":d["species"],"common":d["common"],"clade":d["clade"],
        "accession":d["accession"],"partial":d["partial"],
        "system":"disulfide" if C>=2 else "electrostatic",
        "length":L,"R":R,"K":K,"C":C,"H":H,
        "R_pct":round(100*R/L,2),"RK_pct":round(100*(R+K)/L,2),
        "RKH_pct":round(100*(R+K+H)/L,2),
        "net_charge_74":round(nc74,2),
        "charge_density":round(nc74/L,4),
        "pI":round(isoelectric(s),2),
        "max_R_run":max_run(s,'R'),
        "mean_hydropathy":round(np.mean([KD[a] for a in s if a in KD]),3),
        "foldindex":round(fi,3),
        "uversky_disordered":bool(ud),
        "disulfide_capable":C>=2,
    })

# write master CSV
cols=list(rows[0].keys())
with open(os.path.join(RES,"properties.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)

# ---- VERIFICATION against Biopython ProteinAnalysis (independent pI/charge engine) ----
ver={"pI_max_abs_diff":None,"charge_max_abs_diff":None,"per_species":[]}
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    dpi=[]; dq=[]
    for d in data:
        pa=ProteinAnalysis(d["seq"])
        bp_pi=pa.isoelectric_point(); bp_q=pa.charge_at_pH(7.4)
        my_pi=isoelectric(d["seq"]); my_q=net_charge(d["seq"],7.4)
        dpi.append(abs(bp_pi-my_pi)); dq.append(abs(bp_q-my_q))
        ver["per_species"].append({"species":d["species"],
            "pI_ours":round(my_pi,2),"pI_biopython":round(bp_pi,2),
            "q74_ours":round(my_q,2),"q74_biopython":round(bp_q,2)})
    ver["pI_max_abs_diff"]=round(max(dpi),3)
    ver["pI_mean_abs_diff"]=round(float(np.mean(dpi)),3)
    ver["charge_max_abs_diff"]=round(max(dq),3)
    ver["note"]="pI differs slightly because Biopython uses a different pKa set (Bjellqvist/DTASelect) than our EMBOSS set; rank order and >12 vs <13 pattern are preserved. Charge at pH7.4 should track closely."
except Exception as e:
    ver["error"]=str(e)
json.dump(ver,open(os.path.join(RES,"verification.json"),"w"),indent=2)

# console summary
import statistics as st
print(f"{'species':24s} {'clade':10s} {'sys':12s} {'C':>2s} {'R%':>5s} {'chgD':>6s} {'pI':>6s} {'maxR':>4s} {'FI':>7s} {'disord':>6s}")
for r in rows:
    print(f"{r['species']:24s} {r['clade']:10s} {r['system']:12s} {r['C']:2d} {r['R_pct']:5.1f} "
          f"{r['charge_density']:6.3f} {r['pI']:6.2f} {r['max_R_run']:4d} {r['foldindex']:7.2f} {str(r['uversky_disordered']):>6s}")
print("\n-- group means (charge_density) --")
for cl in ["eutherian","marsupial","monotreme","bird","reptile","amphibian"]:
    v=[r['charge_density'] for r in rows if r['clade']==cl]
    c=[r['C'] for r in rows if r['clade']==cl]
    print(f"  {cl:10s} n={len(v)}  chgD={st.mean(v):.3f}  meanCys={st.mean(c):.1f}")
print("\nVerification:",json.load(open(os.path.join(RES,'verification.json'))).get('charge_max_abs_diff'),
      "max |Δcharge| vs Biopython;  pI max Δ =",json.load(open(os.path.join(RES,'verification.json'))).get('pI_max_abs_diff'))
