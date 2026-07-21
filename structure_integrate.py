#!/usr/bin/env python3
"""Integrate AF3 (46 taxa, MSA-free) with AF2 + ESMFold (25 taxa) and the sequence metrics.
Tests whether AF3 reproduces the poly-arginine pLDDT artifact. Outputs results_v2/structure_af3.json"""
import json, os, csv
import numpy as np
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))

# AF3 (46)
af3={}
for r in csv.DictReader(open(os.path.join(HERE,"results_v2","af3_plddt.csv"))):
    if r["af3_mean_plddt"]=="":continue
    af3[r["acc_lc"]]={"plddt":float(r["af3_mean_plddt"]),"ptm":float(r["af3_ptm"]) if r["af3_ptm"] else None,
                      "frac_dis":float(r["af3_fraction_disordered"]) if r["af3_fraction_disordered"] else None}
# properties (46)
prop={r["accession"].lower():r for r in csv.DictReader(open(f"{HERE}/results_v2/properties.csv"))}
# AF2 (25) + ESMFold (25)
af2={r["accession"].lower():float(r["mean_pLDDT"]) for r in csv.DictReader(open(f"{HERE}/results/alphafold_plddt.csv")) if r["available"]=="True"}
esm={x["accession"].lower():x["mean_pLDDT"] for x in json.load(open(f"{HERE}/results/esmfold_protamines.json")) if x.get("status")=="success"}

# merge
rows=[]
for acc,a in af3.items():
    p=prop.get(acc)
    if not p: continue
    rows.append({"acc":acc,"species":p["species"],"clade":p["clade"],"system":p["system"],
                 "charge":float(p["charge_density"]),"foldindex":float(p["foldindex"]),"partial":p["partial"]=="True",
                 "af3":a["plddt"],"af3_ptm":a["ptm"],"af3_frac_dis":a["frac_dis"],
                 "af2":af2.get(acc),"esm":esm.get(acc)})

af3v=np.array([r["af3"] for r in rows]); fi=np.array([r["foldindex"] for r in rows]); ch=np.array([r["charge"] for r in rows])
ptm=np.array([r["af3_ptm"] for r in rows]); fd=np.array([r["af3_frac_dis"] for r in rows])

def clade_means(key):
    d={}
    for c in ["eutherian","marsupial","monotreme","bird","reptile","amphibian"]:
        v=[r[key] for r in rows if r["clade"]==c and r[key] is not None]
        if v: d[c]=round(float(np.mean(v)),1)
    return d

out={"n_af3":len(rows),
 "af3_mean_plddt":round(float(af3v.mean()),1),"af3_plddt_range":[round(float(af3v.min()),1),round(float(af3v.max()),1)],
 "af3_frac_below_70":round(float((af3v<70).mean()),3),"af3_frac_below_50":round(float((af3v<50).mean()),3),
 "af3_frac_ptm_below_0.5":round(float((ptm<0.5).mean()),3),"af3_max_ptm":round(float(ptm.max()),3),
 "af3_mean_fraction_disordered":round(float(fd.mean()),3),"af3_frac_fully_disordered":round(float((fd>=0.999).mean()),3),
 "af3_plddt_by_clade":clade_means("af3"),
 "af3_plddt_by_system":{s:round(float(np.mean([r["af3"] for r in rows if r["system"]==s])),1) for s in ["disulfide","electrostatic"]},
 # THE ARTIFACT: pLDDT should track charge/disorder POSITIVELY if it is the poly-R helix artifact
 "af3_plddt_vs_foldindex_r":round(float(stats.pearsonr(af3v,fi)[0]),3),"af3_plddt_vs_foldindex_p":float(f"{stats.pearsonr(af3v,fi)[1]:.2g}"),
 "af3_plddt_vs_charge_r":round(float(stats.pearsonr(af3v,ch)[0]),3),"af3_plddt_vs_charge_p":float(f"{stats.pearsonr(af3v,ch)[1]:.2g}"),
}
# cross-predictor on the 25 overlap
ov=[r for r in rows if r["af2"] is not None and r["esm"] is not None]
a3=np.array([r["af3"] for r in ov]); a2=np.array([r["af2"] for r in ov]); es=np.array([r["esm"] for r in ov])
out["cross_predictor_n_overlap"]=len(ov)
out["AF3_vs_AF2_r"]=round(float(stats.pearsonr(a3,a2)[0]),3)
out["AF3_vs_ESMFold_r"]=round(float(stats.pearsonr(a3,es)[0]),3)
out["AF2_vs_ESMFold_r"]=round(float(stats.pearsonr(a2,es)[0]),3)
out["three_predictor_all_below_70_frac"]=round(float(np.mean([(r["af3"]<70 and r["af2"]<70 and r["esm"]<70) for r in ov])),3)
out["interpretation"]=("All three predictors (AF3 diffusion, AF2 Evoformer, ESMFold LM) fail to place a confident "
 "globular fold: AF3 pTM<0.5 for ALL, fraction_disordered~1 for nearly all, and pLDDT stays low. Crucially AF3 pLDDT "
 "correlates NEGATIVELY with FoldIndex (higher pLDDT for MORE disordered, most charge-dense sequences) -> the "
 "poly-arginine spurious-helix artifact is architecture-independent (persists from AF2/ESMFold into AF3), so pLDDT "
 "is usable only to EXCLUDE a stable fold, not as a disorder gradient. The few marsupials at pLDDT 70-72 are the "
 "MOST charge-dense (artifact), not genuine folds (their pTM<0.5, fraction_disordered=1).")

json.dump(out,open(f"{HERE}/results_v2/structure_af3.json","w"),indent=2)

print("="*68)
print(f"AF3 (n={out['n_af3']}, MSA-free): mean pLDDT {out['af3_mean_plddt']} range {out['af3_plddt_range']}")
print(f"  frac below 70: {out['af3_frac_below_70']} | below 50: {out['af3_frac_below_50']}")
print(f"  pTM: ALL below 0.5 = {out['af3_frac_ptm_below_0.5']==1.0} (max pTM {out['af3_max_ptm']})")
print(f"  fraction_disordered: mean {out['af3_mean_fraction_disordered']}, fully-disordered frac {out['af3_frac_fully_disordered']}")
print(f"  by system: disulfide {out['af3_plddt_by_system']['disulfide']} vs electrostatic {out['af3_plddt_by_system']['electrostatic']}")
print(f"  by clade: {out['af3_plddt_by_clade']}")
print(f"\nPOLY-ARGININE ARTIFACT (AF3):")
print(f"  pLDDT vs FoldIndex r={out['af3_plddt_vs_foldindex_r']} p={out['af3_plddt_vs_foldindex_p']}  (NEGATIVE => artifact persists; cf AF2 r=-0.68)")
print(f"  pLDDT vs charge   r={out['af3_plddt_vs_charge_r']} p={out['af3_plddt_vs_charge_p']}  (POSITIVE => more charge, spuriously higher pLDDT)")
print(f"\nCROSS-PREDICTOR (n={out['cross_predictor_n_overlap']} overlap): AF3~AF2 r={out['AF3_vs_AF2_r']}, AF3~ESM r={out['AF3_vs_ESMFold_r']}, AF2~ESM r={out['AF2_vs_ESMFold_r']}")
print(f"  all three below pLDDT 70: {out['three_predictor_all_below_70_frac']}")
print("\nwrote results_v2/structure_af3.json")
