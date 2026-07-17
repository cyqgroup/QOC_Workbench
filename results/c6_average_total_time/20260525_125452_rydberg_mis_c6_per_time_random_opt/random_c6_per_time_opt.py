#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import csv,json,random,sys,time,shutil
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qoc_lbi.protocol import ProtocolCandidate
from qoc_lbi.rydberg_mis import evaluate_rydberg_mis
from qoc_lbi.task_loader import load_task_spec
BASE_TASK=ROOT/'task_specs'/'rydberg_mis_c6.yaml'; OUT=ROOT/'artifacts'
TIMES=[8.0,10.0,12.0,14.0,16.0,18.0,20.0]
BASE=[0.7624613046646118,0.9026690721511841,0.8591732978820801,0.8239520192146301,0.8936153650283813,0.964096188545227,0.9674056768417358]
BASE_MEAN=sum(BASE)/len(BASE); TARGET=1.05*BASE_MEAN
random.seed(20260525)
def ch(n,b,p=None): return {'name':n,'basis':b,'params':p or {}}
def cd_none(): return {'kind':'none','ansatz':None,'order':None,'params':{}}
def cd_scaled(a): return {'kind':'acqc_j0_scaled','ansatz':None,'order':'first_order','params':{'alpha':a}}
def cd_var(vals): return {'kind':'y_sum_parameterized','ansatz':None,'order':'first_order','params':{'basis':'piecewise_linear','scale':1.0,'knots':[0,0.25,0.5,0.75,1],'values':vals}}
def make(label,T,o,d,cd):
 return ProtocolCandidate.from_dict({'candidate_id':f'{label}_T{str(T).replace(".","p")}','task_id':'rydberg_mis_c6','family':'per_time_random_opt','hardware':'rydberg','total_time':T,'channels':[o,d],'cd':cd,'constraints':{'omega_nonnegative':True,'interaction_v_time_independent':True},'provenance':{'source':'codex_random_c6_per_time'},'notes':['per-total-time random protocol optimization','V fixed']},task_id='rydberg_mis_c6')
def candidate_pool_for_T(T):
 pool=[]
 # alpha grid including known best around each T.
 for a in [0.45,0.52,0.6,0.7,0.85,1.0,1.04]: pool.append(make(f'alpha_{a}',T,ch('omega','smooth',{}),ch('delta','smooth',{}),cd_scaled(a)))
 # smooth with variational Y
 for mid in [-0.12,-0.08,-0.04,0.04,0.08,0.12,0.16]: pool.append(make(f'smooth_varY_{mid}',T,ch('omega','smooth',{}),ch('delta','smooth',{}),cd_var([0,0.4*mid,mid,0.4*mid,0])))
 # structured piecewise random-ish; constraints: omega endpoints 0, delta endpoints -1/1.
 knots=[0,0.25,0.5,0.75,1]
 base_shapes=[]
 for _ in range(45):
  o1=random.uniform(0.35,1.0); o2=random.uniform(0.55,1.0); o3=random.uniform(0.2,0.9)
  d1=random.uniform(-0.85,-0.25); d2=random.uniform(-0.15,0.25); d3=random.uniform(0.35,1.0)
  # enforce smooth-ish step <= constraints roughly over 128 samples ok if slopes not too huge, value range ok.
  ymid=random.choice([0, random.uniform(-0.12,0.12)])
  y=[0,0.4*ymid,ymid,0.4*ymid,0]
  base_shapes.append(([0,o1,o2,o3,0],[-1,d1,d2,d3,1],y))
 # add targeted T14/16 shapes with delayed crossing and late cleanup
 targeted=[([0,0.7,1,0.6,0],[-1,-0.65,0.0,0.75,1],[0,0.02,0.05,0.02,0]),([0,0.8,1,0.45,0],[-1,-0.75,-0.05,0.65,1],[0,0,0,0,0]),([0,0.55,1,0.75,0],[-1,-0.45,0.05,0.85,1],[0,-0.02,-0.05,-0.02,0])]
 base_shapes += targeted
 for i,(ov,dv,y) in enumerate(base_shapes):
  pool.append(make(f'pw_rand_{i}',T,ch('omega','piecewise_linear',{'knots':knots,'values':ov}),ch('delta','piecewise_linear',{'knots':knots,'values':dv}),cd_none() if max(abs(v) for v in y)<1e-12 else cd_var(y)))
 return pool
def eval_c(spec,c):
 try:
  r=evaluate_rydberg_mis(spec,c,'full'); f=np.asarray(r['artifact_bundle']['target_fidelity']); return float(f[-1]),''
 except Exception as e: return None,repr(e)
def main():
 spec=load_task_spec(BASE_TASK); run=OUT/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rydberg_mis_c6_per_time_random_opt"; run.mkdir(parents=True)
 allrows=[]; chosen=[]
 for T,base in zip(TIMES,BASE):
  best={'T':T,'candidate_id':'baseline','final_fidelity':base,'error':'','candidate':None}
  pool=candidate_pool_for_T(T)
  for i,c in enumerate(pool):
   st=time.time(); final,err=eval_c(spec,c); row={'T':T,'candidate_id':c.candidate_id,'final_fidelity':final,'error':err,'elapsed_s':time.time()-st,'candidate':c.to_dict()}; allrows.append(row)
   if final is not None and final>best['final_fidelity']: best={'T':T,'candidate_id':c.candidate_id,'final_fidelity':final,'error':'','candidate':c.to_dict()}
   if i%20==0: print(f'T={T} i={i}/{len(pool)} current_best={best["final_fidelity"]}',flush=True)
  chosen.append(best); print(f'[BEST] T={T} {best["candidate_id"]} final={best["final_fidelity"]}',flush=True)
 mean=float(np.mean([r['final_fidelity'] for r in chosen])); gain=mean/BASE_MEAN-1
 with (run/'all_candidates.csv').open('w',newline='') as f:
  fields=['T','candidate_id','final_fidelity','error','elapsed_s']; w=csv.DictWriter(f,fields); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in allrows]
 with (run/'chosen_by_time.csv').open('w',newline='') as f:
  fields=['T','candidate_id','final_fidelity','error']; w=csv.DictWriter(f,fields); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in chosen]
 (run/'chosen_protocols.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
 plt.figure(figsize=(7,4.5)); plt.plot(TIMES,BASE,'--o',label=f'baseline mean={BASE_MEAN:.3f}'); plt.plot(TIMES,[r['final_fidelity'] for r in chosen],'-o',label=f'per-T optimized mean={mean:.3f}'); plt.axhline(TARGET,ls=':',color='gray',label='5% target mean'); plt.ylim(0,1.02); plt.xlabel('total annealing time T'); plt.ylabel('final fidelity'); plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(run/'final_fidelity_vs_total_time.png',dpi=180); plt.close()
 (run/'index.html').write_text(f"<html><meta charset='utf-8'><body><h1>C6 per-time optimized average final fidelity</h1><p>baseline mean={BASE_MEAN:.12f}; target={TARGET:.12f}; optimized mean={mean:.12f}; gain={100*gain:.2f}%</p><img src='final_fidelity_vs_total_time.png'></body></html>")
 shutil.copy2(Path(__file__),run/'random_c6_per_time_opt.py')
 print(json.dumps({'run_dir':str(run),'baseline_mean':BASE_MEAN,'target':TARGET,'mean':mean,'gain':gain,'chosen':chosen},ensure_ascii=False,indent=2),flush=True)
 if mean<TARGET: raise RuntimeError('target not reached')
if __name__=='__main__': main()
