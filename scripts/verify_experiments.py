"""Audit frozen M7 artifacts without training; --prepare builds derived evidence.

Run from any directory with the project's Python environment. Checkpoints are
loaded with weights_only=True. Existing metric values are never silently replaced.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import math
import platform
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from source.predict import load_run_config, load_model
from source.dataset import HorseHumanDataset
from source.preprocessing import build_preprocessing
from source.trainer import evaluate

RUNS = dict(E1='E1_relu_normal_aug', E2='E2_relu_xavier_aug', E3='E3_relu_he_aug',
            E4='E4_sigmoid_xavier_aug', E5='E5_relu_he_no_aug')
EXPECTED = [('relu','normal',True),('relu','xavier_normal',True),('relu','he_normal',True),
            ('sigmoid','xavier_normal',True),('relu','he_normal',False)]
COLORS = dict(zip(RUNS, ['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']))
FIGURES = ['initialization_train_loss','initialization_val_loss','activation_comparison',
           'augmentation_effect','validation_metrics_comparison','best_model_training_curves',
           'test_confusion_matrix','test_predictions']
METRICS = ROOT/'results/metrics'
FIGDIR = ROOT/'results/figures'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-7)


def check_metrics(s,n,positive):
    for k in ('loss','accuracy','precision','recall','f1','tp','tn','fp','fn'):
        require(math.isfinite(s[k]),f'Nonfinite {k}')
    require(s['loss']>=0,'Negative loss')
    tp,tn,fp,fn=[s[k] for k in ('tp','tn','fp','fn')]
    require(all(v>=0 and int(v)==v for v in (tp,tn,fp,fn)), 'Invalid counts')
    require(tp+tn+fp+fn==n and tp+fn==positive,'Wrong sample/class totals')
    expected=dict(accuracy=(tp+tn)/n,precision=tp/(tp+fp) if tp+fp else 0,
                  recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    for k,v in expected.items():
        require(0<=s[k]<=1 and close(s[k],v),f'{k} inconsistent with confusion counts')


def audit_runs():
    histories={}; configs={}; rows=[]
    fixed=dict(seed=42,image_size=[64,64],image_channels=3,num_classes=2,hidden_dims=[128,64],
               batch_size=32,learning_rate=.001,epochs=30,optimizer='Adam',loss='BCEWithLogitsLoss',num_workers=0,device='cpu')
    for (eid,run),protocol in zip(RUNS.items(),EXPECTED):
        c=read(ROOT/f'results/configs/{run}.json'); h=read(ROOT/f'results/histories/{run}.json')
        require(all(c.get(k)==v for k,v in fixed.items()),f'{eid}: fixed protocol mismatch')
        require(tuple(c[k] for k in ('activation','initialization','augmentation_enabled'))==protocol,f'{eid}: variable protocol')
        require(c['run_name']==run,f'{eid}: run_name')
        for split,n,pos in [('train',845,434),('val',182,93)]:
            require(len(h[split])==30 and [s['epoch'] for s in h[split]]==list(range(1,31)),f'{eid}: epochs')
            for s in h[split]: check_metrics(s,n,pos)
        best=min(h['val'],key=lambda s:s['loss']); epoch=best['epoch']
        require(h['best_epoch']==epoch and close(h['best_val_loss'],best['loss']),f'{eid}: best epoch/loss')
        cp=ROOT/f'results/checkpoints/{run}.pt'
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        require(ck['epoch']==epoch and close(ck['best_val_loss'],best['loss']),f'{eid}: checkpoint metadata')
        require('optimizer_state_dict' in ck,'Missing optimizer metadata')
        require(all(torch.isfinite(v).all() for v in ck['model_state_dict'].values()),f'{eid}: invalid weights')
        cfg=load_run_config(run,device='cpu')
        # Strict shape/key compatibility, without initializing trained weights again.
        model=load_model(cp,cfg)
        require(sum(p.numel() for p in model.parameters())==1581313,f'{eid}: architecture')
        rows.append(dict(experiment=eid,run_name=run,activation=c['activation'],initialization=c['initialization'],
                         augmentation=c['augmentation_enabled'],best_epoch=epoch,train_loss=h['train'][epoch-1]['loss'],
                         validation_loss=best['loss'],**{k:best[k] for k in ('accuracy','precision','recall','f1','tp','tn','fp','fn')}))
        histories[eid]=h; configs[eid]=c
    require(min(rows,key=lambda r:r['validation_loss'])['experiment']=='E5','Validation no longer selects E5; stop, do not reselect by Test')
    return histories,configs,rows


@torch.no_grad()
def prepare():
    histories,configs,rows=audit_runs()
    # Preserve existing comparison measurements while enriching missing protocol columns.
    old_path=METRICS/'experiment_comparison.json'
    if old_path.exists():
        for old in read(old_path):
            row=next(r for r in rows if r['run_name']==old['run_name'])
            for k in ('best_epoch','accuracy','precision','recall','f1','tp','tn','fp','fn'):
                if k in old: require(close(row[k],old[k]),f'Existing comparison mismatch: {k}')
            require(close(row['validation_loss'],old.get('validation_loss',old.get('loss'))),'Existing comparison loss mismatch')
    frame=pd.DataFrame(rows)
    frame.to_csv(METRICS/'experiment_comparison.csv',index=False)
    (METRICS/'experiment_comparison.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    restored={}
    # Verify frozen checkpoint predictions on Validation, not retraining.
    for eid,run in RUNS.items():
        cfg=load_run_config(run,device='cpu')
        ds=HorseHumanDataset(cfg.splits_dir/'val.csv',build_preprocessing(cfg.image_size))
        stats=evaluate(load_model(ROOT/f'results/checkpoints/{run}.pt',cfg),DataLoader(ds,batch_size=32),torch.nn.BCEWithLogitsLoss(),'cpu')
        best=histories[eid]['val'][histories[eid]['best_epoch']-1]
        require(all((math.isclose(stats[k],best[k],rel_tol=1e-5,abs_tol=1e-5) if k=='loss' else close(stats[k],best[k])) for k in stats),f'{eid}: restored Validation mismatch')
        restored[eid]=stats
    run=RUNS['E5']; cfg=load_run_config(run,device='cpu'); cp=ROOT/f'results/checkpoints/{run}.pt'
    ds=HorseHumanDataset(cfg.splits_dir/'test.csv',build_preprocessing(cfg.image_size))
    model=load_model(cp,cfg) # One load for all Test batches.
    records=[]; losses=[]; offset=0
    for images,targets in DataLoader(ds,batch_size=32,shuffle=False):
        logits=model(images); probabilities=torch.sigmoid(logits)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='sum').item())
        for z,p,y in zip(logits.flatten().tolist(),probabilities.flatten().tolist(),targets.flatten().tolist()):
            label=int(p>=.5); y=int(y)
            records.append(dict(image_path=ds.samples.iloc[offset].image_path,ground_truth=y,predicted_label=label,
                                human_probability=p,logit=z,status='Correct' if label==y else 'Incorrect'))
            offset+=1
    predictions=pd.DataFrame(records)
    y=predictions.ground_truth.to_numpy(); pred=predictions.predicted_label.to_numpy()
    tp=int(((y==1)&(pred==1)).sum()); tn=int(((y==0)&(pred==0)).sum())
    fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    actual=dict(loss=sum(losses)/len(ds),accuracy=(tp+tn)/len(ds),precision=tp/(tp+fp),recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),tp=tp,tn=tn,fp=fp,fn=fn)
    finalpath=METRICS/'final_test_metrics.json'
    final=read(finalpath) if finalpath.exists() else dict(run_name=run,best_epoch=histories['E5']['best_epoch'],split='test',**actual)
    require(final['run_name']==run and final['best_epoch']==histories['E5']['best_epoch'],'Final model mismatch')
    require(all(close(final[k],v) for k,v in actual.items()),'Existing final metrics differ from actual inference; not overwriting')
    final.update(model_configuration={k:v for k,v in configs['E5'].items() if k!='root'},
                 checkpoint_path=str(cp.relative_to(ROOT)),number_of_samples=len(ds),accuracy_percentage=100*final['accuracy'],classification_threshold=.5)
    finalpath.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    predictions.to_csv(METRICS/'test_prediction_details.csv',index=False)
    selected=draw_figures(histories,rows,final,predictions)
    inputs=[ROOT/f'results/{folder}/{run}.{ext}' for run in RUNS.values() for folder,ext in [('configs','json'),('histories','json'),('checkpoints','pt')]]
    inputs += list((ROOT/'data/splits').glob('*.csv'))
    inputs += [ROOT/f'source/{name}.py' for name in ['model','train','trainer','predict','metrics','dataset','preprocessing','augmentation']]
    generated=[METRICS/name for name in ['experiment_comparison.csv','experiment_comparison.json','final_test_metrics.json','test_prediction_details.csv']]+[FIGDIR/f'{name}.png' for name in FIGURES]
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),torch=torch.__version__,
                  inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},outputs={str(p.relative_to(ROOT)):digest(p) for p in generated},
                  restored_validation=restored,restored_loss_tolerance={'relative':1e-5,'absolute':1e-5,'reason':'float32 reductions can vary with CPU threading; confusion counts must match'},selected_prediction_paths=selected,selection_seed=42,
                  training_runtime_versions='Not recorded per historical run; versions above describe verification only',
                  historical_augmentation_evidence='Saved flags + current CLI train/eval transform routing + regression, not a historical execution trace')
    (METRICS/'m7_artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return rows,final


def draw_figures(histories,rows,final,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','font.size':11})
    def save(fig,name):
        fig.savefig(FIGDIR/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close(fig)
    def curve(ids,split,name,title):
        fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
        for eid in ids:
            h=histories[eid][split]
            ax.plot([s['epoch'] for s in h],[s['loss'] for s in h],color=COLORS[eid],label=RUNS[eid],marker='.',lw=1.8)
        ax.set(title=title,xlabel='Epoch',ylabel='Binary cross entropy loss',yscale='log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend(); ax.grid(alpha=.25,which='both')
        save(fig,name)
    curve(['E1','E2','E3'],'train','initialization_train_loss','Initialization: training convergence (log loss)')
    curve(['E1','E2','E3'],'val','initialization_val_loss','Initialization: validation loss')
    curve(['E2','E4'],'train','activation_comparison','ReLU vs Sigmoid — Xavier initialization, augmentation on')
    curve(['E3','E5'],'val','augmentation_effect','Augmentation — ReLU + He, validation loss')
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['accuracy','precision','recall','f1']):
        ax.bar([r['experiment'] for r in rows],[r[metric] for r in rows],color=[COLORS[r['experiment']] for r in rows])
        ax.set(title=metric.capitalize(),ylim=(0,1.05),ylabel='Score'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation metrics at each run’s minimum-validation-loss checkpoint')
    save(fig,'validation_metrics_comparison')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,metric in zip(axes.flat,['loss','accuracy','precision','recall','f1']):
        for split,style in [('train','-'),('val','--')]:
            ax.plot(range(1,31),[r[metric] for r in histories['E5'][split]],style,color=COLORS['E5'] if split=='train' else '#333333',label=split.capitalize())
        ax.axvline(histories['E5']['best_epoch'],color='#0072B2',ls=':',label=f"Best epoch {histories['E5']['best_epoch']}")
        ax.set(title=metric.capitalize(),xlabel='Epoch',ylabel=metric); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes.flat[-1].axis('off'); fig.suptitle('E5 — final model selected using Validation only')
    save(fig,'best_model_training_curves')
    matrix=np.array([[final['tn'],final['fp']],[final['fn'],final['tp']]],dtype=int)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained'); im=ax.imshow(matrix,cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=18,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Horse (0)','Human (1)'],yticklabels=['Horse (0)','Human (1)'],xlabel='Predicted Label',ylabel='True Label',title='E5 Test — threshold 0.5, n=256'); fig.colorbar(im,ax=ax)
    save(fig,'test_confusion_matrix')
    rng=np.random.default_rng(42); selected={}
    fig,axes=plt.subplots(4,4,figsize=(13,14),layout='constrained')
    for row,(group,y,p) in enumerate([('TN',0,0),('TP',1,1),('FP',0,1),('FN',1,0)]):
        pool=details[(details.ground_truth==y)&(details.predicted_label==p)]
        chosen=rng.choice(pool.index.to_numpy(),size=min(4,len(pool)),replace=False)
        selected[group]=details.loc[chosen,'image_path'].tolist()
        for col,ax in enumerate(axes[row]):
            ax.axis('off')
            if col>=len(chosen): continue
            r=details.loc[chosen[col]]
            with Image.open(ROOT/r.image_path) as img: ax.imshow(img.convert('RGB'))
            ax.set_title(f"{group} | GT: {'Human' if y else 'Horse'} → Pred: {'Human' if p else 'Horse'}\nP(Human)={r.human_probability:.4f} | {r.status}",fontsize=9,color='#14733b' if y==p else '#a12424')
    fig.suptitle('E5 Test examples — seeded selection (42), four per confusion group')
    save(fig,'test_predictions')
    return selected


def verify(notebook=True):
    checks={}
    def check(name,fn):
        try: fn(); checks[name]={'status':'PASS'}
        except Exception as exc: checks[name]={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'}
    state={}
    def experiments():
        h,c,r=audit_runs(); state.update(histories=h,configs=c,rows=r)
    check('five_runs_protocol_30_epochs_metrics_best_checkpoint',experiments)
    def comparison():
        expected=pd.DataFrame(state['rows'])
        for path in ['experiment_comparison.json','experiment_comparison.csv']:
            actual=pd.DataFrame(read(METRICS/path)) if path.endswith('json') else pd.read_csv(METRICS/path)
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=1e-9,atol=1e-9)
    check('comparison_same_best_epoch',comparison)
    def final_metrics():
        f=read(METRICS/'final_test_metrics.json'); check_metrics(f,256,128)
        require(f['run_name']==RUNS['E5'] and f['split']=='test' and f['best_epoch']==state['histories']['E5']['best_epoch'],'Final selection metadata')
        require(f['classification_threshold']==.5 and f['number_of_samples']==256 and close(f['accuracy_percentage'],100*f['accuracy']),'Final metadata')
        require(f['checkpoint_path']==f"results/checkpoints/{RUNS['E5']}.pt",'Checkpoint path')
        require(f['model_configuration']=={k:v for k,v in state['configs']['E5'].items() if k!='root'},'Model config')
        state['final']=f
    check('final_test_metrics_confusion_consistency',final_metrics)
    def predictions():
        d=pd.read_csv(METRICS/'test_prediction_details.csv'); split=pd.read_csv(ROOT/'data/splits/test.csv')
        require(len(d)==256 and d.image_path.is_unique,'Prediction count/duplicates')
        require(d.image_path.tolist()==split.image_path.tolist() and d.ground_truth.tolist()==split.label.tolist(),'Prediction membership/targets')
        require(np.isfinite(d[['human_probability','logit']]).all().all() and d.human_probability.between(0,1).all(),'Invalid predictions')
        require((d.predicted_label==(d.human_probability>=.5).astype(int)).all(),'Threshold')
        require(np.allclose(d.human_probability,torch.sigmoid(torch.tensor(d.logit.to_numpy())).numpy(),atol=1e-7),'Probability vs logits')
        correct=d.ground_truth==d.predicted_label
        require(d.status.tolist()==np.where(correct,'Correct','Incorrect').tolist(),'Correct status')
        f=state['final']; y=d.ground_truth;p=d.predicted_label
        for key,a,b in [('tn',0,0),('tp',1,1),('fp',0,1),('fn',1,0)]: require(int(((y==a)&(p==b)).sum())==f[key],key)
        loss=np.mean(np.maximum(d.logit,0)-d.logit*d.ground_truth+np.log1p(np.exp(-np.abs(d.logit))))
        require(close(loss,f['loss']),'Loss from logits')
    check('all_test_prediction_details',predictions)
    def selection():
        d=pd.read_csv(METRICS/'test_prediction_details.csv')
        m=read(METRICS/'m7_artifact_manifest.json'); rng=np.random.default_rng(42)
        for group,y,p in [('TN',0,0),('TP',1,1),('FP',0,1),('FN',1,0)]:
            pool=d[(d.ground_truth==y)&(d.predicted_label==p)]
            chosen=rng.choice(pool.index.to_numpy(),size=min(4,len(pool)),replace=False)
            require(d.loc[chosen,'image_path'].tolist()==m['selected_prediction_paths'][group],f'{group}: seed selection mismatch')
    check('seed42_four_confusion_groups',selection)
    def manifest():
        m=read(METRICS/'m7_artifact_manifest.json')
        for mapping in ['inputs','outputs']:
            for path,sha in m[mapping].items(): require(digest(ROOT/path)==sha,f'Artifact changed: {path}')
        for eid,s in m['restored_validation'].items():
            best=state['histories'][eid]['val'][state['histories'][eid]['best_epoch']-1]
            require(all((math.isclose(s[k],best[k],rel_tol=1e-5,abs_tol=1e-5) if k=='loss' else close(s[k],best[k])) for k in s),'Restored validation')
    check('provenance_hashes_and_restored_validation',manifest)
    def figures():
        for name in FIGURES:
            with Image.open(FIGDIR/f'{name}.png') as img:
                require(min(img.info.get('dpi',(0,0)))>=199.9,f'{name}: DPI')
                require(np.asarray(img.convert('RGB')).std()>5,f'{name}: blank')
                img.load()
    check('eight_readable_nonblank_figures_200dpi',figures)
    if notebook:
        def nb():
            n=read(ROOT/'source/experiments/06_comparative_experiments.ipynb')
            require(not any('TODO' in ''.join(c['source']) for c in n['cells']),'Remaining TODO')
            code=[c for c in n['cells'] if c['cell_type']=='code']
            require(code and all(c.get('execution_count') is not None for c in code),'Unexecuted cells')
            require(not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[])),'Notebook errors')
            require(any('image/png' in o.get('data',{}) for c in code for o in c.get('outputs',[])),'No saved figures')
        check('notebook_run_all_saved_outputs',nb)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='Generate derived evidence; never train')
    parser.add_argument('--regression-xml',type=Path)
    args=parser.parse_args(); torch.set_num_threads(min(torch.get_num_threads(),2))
    if args.prepare:
        prepare(); print('Derived evidence prepared; run notebook then verify.'); return
    checks=verify(); regression=None
    if args.regression_xml:
        suites=ET.parse(args.regression_xml).findall('.//testsuite')
        regression={k:sum(int(s.get(k,'0')) for s in suites) for k in ['tests','failures','errors','skipped']}
        regression['passed']=regression['tests']-sum(regression[k] for k in ['failures','errors','skipped'])
        regression['duration_seconds']=sum(float(s.get('time',0)) for s in suites)
        checks['regression']={'status':'PASS' if regression['tests']>0 and not regression['failures'] and not regression['errors'] and not regression['skipped'] else 'FAIL'}
    else: checks['regression']={'status':'NOT RUN'}
    result=dict(date=datetime.now(timezone.utc).isoformat(),checks=checks,regression=regression,
                experiments=read(METRICS/'experiment_comparison.json'),final_test=read(METRICS/'final_test_metrics.json'),
                technical_done=all(c['status']=='PASS' for c in checks.values()),
                artifacts=[str(p.relative_to(ROOT)) for p in [METRICS/n for n in ['experiment_comparison.csv','experiment_comparison.json','final_test_metrics.json','test_prediction_details.csv','m7_artifact_manifest.json','m7_verification_summary.json','m7_verification_report.txt']]+[FIGDIR/f'{n}.png' for n in FIGURES]]+['scripts/verify_experiments.py','scripts/execute_m7_notebook.py','source/experiments/06_comparative_experiments.ipynb','PLAN.md'],
                notebook_execution=read(ROOT/'source/experiments/06_comparative_experiments.ipynb').get('metadata',{}).get('m7_execution'),
                limitations=['One seed only; no confidence intervals or repeated training.',
                  'Historical training duration/software versions and execution trace not recorded; verification versions/hashes saved now.',
                  'Train-only augmentation supported by saved config/current CLI routing/regression; no historical per-batch trace.',
                  'No causal attribution to vanishing gradients, overfitting or distribution shift; no Test tuning.',
                  'Checkpoints intentionally ignored by Git; retain locally and package separately per submission policy.',
                  'All notebook cells executed via fresh-process IPython in .venv with rich outputs; standard Jupyter kernel transport is blocked by sandbox socket policy.'])
    (METRICS/'m7_verification_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (METRICS/'m7_verification_report.txt').write_text('M7 ARTIFACT VERIFICATION\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(checks,indent=2)); sys.exit(0 if result['technical_done'] else 1)

if __name__=='__main__': main()
