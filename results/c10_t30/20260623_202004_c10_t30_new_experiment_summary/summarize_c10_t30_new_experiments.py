#!/usr/bin/env python3
from __future__ import annotations
import csv, json, shutil, uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT_ROOT=ROOT/'finalresult/results/c10_t30'
CACHE=OUT_ROOT/'single_candidate_cache'
PARTIAL=OUT_ROOT/'20260623_190840_expanded_c10_t30_hardware_search'/'summary_partial.csv'
BASE=ROOT/'artifacts/20260522_154420_rydberg_mis_c10_smooth_acqc_T30_baseline/trajectory_diagnostics.npz'
PRIOR=ROOT/'finalresult/results/c10_t30/20260522_171157_rydberg_mis_c10_T30_refined_ycd_beyond/trajectory_diagnostics.npz'
REG=ROOT/'registry.jsonl'
BASE_FINAL=0.8724538087844849
NO_Y_FINAL=0.9066373109817505
PRIOR_FINAL=0.9164813756942749

def read_rows():
    rows=[]
    if PARTIAL.exists():
        with PARTIAL.open() as f:
            for r in csv.DictReader(f):
                if r.get('ok')=='True' and r.get('final_fidelity'):
                    rows.append({'candidate_id':r['candidate_id'],'family':r['family'],'final_fidelity':float(r['final_fidelity']),'max_fidelity':float(r['max_fidelity']),'source':'expanded_partial','elapsed_s':float(r['elapsed_s']),'candidate':None})
    for p in sorted(CACHE.glob('*.json')):
        r=json.loads(p.read_text())
        if r.get('ok') and r.get('final_fidelity') is not None:
            rows.append({'candidate_id':r['candidate_id'],'family':r['family'],'final_fidelity':float(r['final_fidelity']),'max_fidelity':float(r['max_fidelity']),'source':'single_focused','elapsed_s':float(r['elapsed_s']),'candidate':r['candidate']})
    # de-duplicate by candidate_id keeping highest precision/newest source preference
    by={}
    for r in rows:
        if r['candidate_id'] not in by or r['source']=='single_focused': by[r['candidate_id']]=r
    return list(by.values())

def save_csv(path, rows):
    fields=['candidate_id','family','source','final_fidelity','max_fidelity','gain_vs_acqc','gain_vs_no_y','gain_vs_prior_refined','elapsed_s']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in sorted(rows,key=lambda x:x['final_fidelity'],reverse=True):
            rr=dict(r); rr['gain_vs_acqc']=r['final_fidelity']/BASE_FINAL-1; rr['gain_vs_no_y']=r['final_fidelity']/NO_Y_FINAL-1; rr['gain_vs_prior_refined']=r['final_fidelity']/PRIOR_FINAL-1
            w.writerow({k:rr.get(k) for k in fields})

def plot(run, rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
    top=sorted(rows,key=lambda r:r['final_fidelity'],reverse=True)[:18]
    fig,ax=plt.subplots(figsize=(7.3,3.7))
    labels=[r['candidate_id'] for r in top][::-1]; vals=[r['final_fidelity'] for r in top][::-1]
    cols=['#d9480f' if v>PRIOR_FINAL else '#3b5bdb' if v>NO_Y_FINAL else '#6c757d' for v in vals]
    ax.barh(range(len(labels)),vals,color=cols)
    ax.axvline(BASE_FINAL,color='#6c757d',ls='--',lw=1,label='ACQC baseline')
    ax.axvline(NO_Y_FINAL,color='#495057',ls=':',lw=1,label='smooth no-Y')
    ax.axvline(PRIOR_FINAL,color='#3b5bdb',ls='--',lw=1,label='prior m=0.13')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_xlabel('final target fidelity'); ax.set_xlim(0.84,0.935); ax.legend(frameon=False,loc='lower right')
    fig.tight_layout(); fig.savefig(run/'c10_top_protocols.pdf'); fig.savefig(run/'c10_top_protocols.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.2))
    fams=sorted(set(r['family'] for r in rows))
    for i,fam in enumerate(fams):
        ys=[r['final_fidelity'] for r in rows if r['family']==fam]
        ax.scatter([i]*len(ys),ys,s=26)
    ax.axhline(BASE_FINAL,color='#6c757d',ls='--',lw=1); ax.axhline(PRIOR_FINAL,color='#3b5bdb',ls='--',lw=1)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams,rotation=30,ha='right'); ax.set_ylabel('final target fidelity')
    fig.tight_layout(); fig.savefig(run/'c10_family_scatter.pdf'); fig.savefig(run/'c10_family_scatter.png',dpi=220); plt.close(fig)
    b=np.load(BASE); p=np.load(PRIOR)
    best=max(rows,key=lambda r:r['final_fidelity']); best_npz=CACHE/f"{best['candidate_id']}.npz"; bn=np.load(best_npz)
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    axs[0].plot(b['times'],b['target_fidelity'],'--',color='#6c757d',label='ACQC baseline')
    axs[0].plot(p['times'],p['target_fidelity'],':',color='#3b5bdb',label='prior m=0.13')
    axs[0].plot(bn['times'],bn['target_fidelity'],'-',color='#d9480f',label='best new')
    axs[0].set_xlabel('time'); axs[0].set_ylabel('target fidelity'); axs[0].set_ylim(0,1); axs[0].legend(frameon=False)
    axs[1].plot(bn['times'],bn['omega_values'],label=r'$\Omega(t)$')
    axs[1].plot(bn['times'],bn['delta_values'],label=r'$\Delta(t)$')
    axs[1].plot(bn['times'],bn['cd_values'],label=r'$f_Y(t)$')
    axs[1].set_xlabel('time'); axs[1].set_ylabel('control'); axs[1].legend(frameon=False)
    for lab,ax in zip('ab',axs): ax.text(-0.15,1.08,lab,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
    fig.tight_layout(); fig.savefig(run/'c10_best_trajectory_controls.pdf'); fig.savefig(run/'c10_best_trajectory_controls.png',dpi=220); plt.close(fig)

def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); t=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run=OUT_ROOT/f'{stamp}_c10_t30_new_experiment_summary'; run.mkdir()
    rows=read_rows(); save_csv(run/'summary.csv',rows); plot(run,rows)
    best=max(rows,key=lambda r:r['final_fidelity'])
    if best['candidate']:
        (run/'best_protocol.json').write_text(json.dumps(best['candidate'],ensure_ascii=False,indent=2))
    shutil.copy2(Path(__file__),run/'summarize_c10_t30_new_experiments.py')
    finding=f"C10/T30 supplemental hardware-feasible experiments found best protocol {best['candidate_id']} with final fidelity {best['final_fidelity']:.12f}, {100*(best['final_fidelity']/BASE_FINAL-1):.2f}% above ACQC baseline and {100*(best['final_fidelity']/PRIOR_FINAL-1):.2f}% above the prior m=0.13 refined pulse."
    (run/'README.md').write_text('# C10 T30 supplemental experiment summary\n\n'+finding+'\n')
    (run/'index.html').write_text(f"<!doctype html><meta charset='utf-8'><body><h1>C10/T30 supplemental experiments</h1><p>{finding}</p><img src='c10_top_protocols.png'><img src='c10_family_scatter.png'><img src='c10_best_trajectory_controls.png'></body>")
    with REG.open('a') as f:
        f.write(json.dumps({'uuid':str(uuid.uuid4()),'time':t,'command':'python qoc_lbi/finalresult/codes/c10_t30/summarize_c10_t30_new_experiments.py','code_path':'qoc_lbi/finalresult/codes/c10_t30/summarize_c10_t30_new_experiments.py','hyperparameter':'Summary of C10/T30 supplemental hardware-feasible global-Y experiments; includes partial expanded sweep and single-candidate focused runs','input_path':[str(PARTIAL.relative_to(ROOT)),str(CACHE.relative_to(ROOT)),str(BASE.relative_to(ROOT)),str(PRIOR.relative_to(ROOT))],'output_path':[str(p.relative_to(ROOT)) for p in sorted(run.iterdir())],'metrics':{'baseline_acqc_final':BASE_FINAL,'smooth_no_y_final':NO_Y_FINAL,'prior_refined_final':PRIOR_FINAL,'best_final_fidelity':best['final_fidelity'],'best_gain_vs_acqc':best['final_fidelity']/BASE_FINAL-1,'best_gain_vs_prior':best['final_fidelity']/PRIOR_FINAL-1,'num_unique_candidates':len(rows)},'finding':finding,'valid':True,'comment':'Summary figure artifact; underlying candidate outputs are cached as JSON/NPZ in single_candidate_cache and partial expanded sweep.'},ensure_ascii=False)+'\n')
    print(json.dumps({'run_dir':str(run),'best':best,'num_rows':len(rows)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
