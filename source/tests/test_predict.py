"""M6 inference and a bounded Train/Validation smoke run; no real Test inference."""
import json
import math
import random
from dataclasses import replace,asdict
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
import source.predict as inference
from source.config import CONFIG
from source.model import MLP,initialize_model
from source.dataset import HorseHumanDataset
from source.preprocessing import build_preprocessing
from source.augmentation import build_augmentation
from source.trainer import fit
from source.visualization import plot_history,plot_confusion_matrix


@pytest.fixture(autouse=True)
def isolate_rng_threads():
    state=random.getstate(); old=torch.get_num_threads()
    torch.set_num_threads(min(old,2))
    with torch.random.fork_rng():
        torch.manual_seed(42); random.seed(42)
        yield
    random.setstate(state); torch.set_num_threads(old)


@pytest.fixture
def saved(tmp_path,monkeypatch):
    cfg=replace(CONFIG,root=tmp_path,image_size=(8,6),hidden_dims=[7,3],activation='sigmoid',device='cpu')
    monkeypatch.setattr(inference,'CONFIG',cfg)
    path=cfg.results_dir/'configs'/'unit.json'; path.parent.mkdir(parents=True)
    data=asdict(cfg); data.update(root='/obsolete/machine',run_name='unit',augmentation_enabled=True,device='cuda')
    path.write_text(json.dumps(data))
    model=MLP(cfg.input_dim,cfg.hidden_dims,cfg.activation)
    cp=tmp_path/'model.pt'
    torch.save({'model_state_dict':model.state_dict(),'epoch':1,'best_val_loss':.5},cp)
    return cfg,model,cp


def test_load_config_portability_fields_and_device_override(saved):
    cfg,_,_=saved
    restored=inference.load_run_config('unit')
    assert restored.root==cfg.root and restored.device=='cpu'
    assert restored.image_size==(8,6) and restored.hidden_dims==[7,3] and restored.activation=='sigmoid'
    assert inference.load_run_config('unit',device='cpu')==restored


def test_load_model_exact_parameters_mode_and_safe_load(saved,monkeypatch):
    cfg,original,cp=saved
    load=Mock(wraps=torch.load); monkeypatch.setattr(torch,'load',load)
    model=inference.load_model(cp,cfg)
    assert load.call_args.kwargs==dict(map_location='cpu',weights_only=True)
    assert not model.training and next(model.parameters()).device.type=='cpu'
    for key,value in model.state_dict().items(): assert torch.equal(value,original.state_dict()[key])
    x=torch.randn(2,3,8,6)
    torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)


@pytest.mark.parametrize('logit,label,name',[(-2.,0,'horse'),(0.,1,'human'),(2.,1,'human')])
def test_prediction_preprocessing_batch_no_grad_and_threshold(logit,label,name,saved,tmp_path,monkeypatch):
    cfg,_,cp=saved
    img=np.zeros((13,11,4),dtype=np.uint8); img[...,2:]=255 # Opaque red in BGRA.
    path=tmp_path/'red.png'; assert cv2.imwrite(str(path),img)
    class Inspect(nn.Module):
        def forward(self,x):
            assert not torch.is_grad_enabled()
            assert x.shape==(1,3,8,6) and x.dtype==torch.float32
            assert torch.all(x[:,0]==1) and torch.all(x[:,1:]==0)
            return torch.tensor([[logit]])
    monkeypatch.setattr(inference,'load_model',Mock(return_value=Inspect().eval()))
    result=inference.predict(path,cp,cfg)
    assert result['label']==label
    assert result['class_name']==name
    assert result['human_probability']==pytest.approx(1/(1+math.exp(-logit)))
    assert 0<=result['human_probability']<=1


@pytest.mark.parametrize('kind',['missing_image','invalid_image','missing_checkpoint','missing_state','wrong_shapes','corrupt_checkpoint'])
def test_missing_and_invalid_files(kind,saved,tmp_path):
    cfg,_,cp=saved
    image=tmp_path/'image.png'
    if kind!='missing_image':
        if kind=='invalid_image': image.write_text('not an image')
        else: assert cv2.imwrite(str(image),np.zeros((8,6,3),np.uint8))
    if kind=='missing_checkpoint': cp=tmp_path/'absent.pt'
    elif kind=='missing_state': torch.save({'epoch':1},cp)
    elif kind=='wrong_shapes': torch.save({'model_state_dict':{'layers.0.weight':torch.zeros(2,2)}},cp)
    elif kind=='corrupt_checkpoint': cp.write_bytes(b'not a torch checkpoint')
    import pickle
    expected={'missing_image':ValueError,'invalid_image':ValueError,'missing_checkpoint':FileNotFoundError,
              'missing_state':KeyError,'wrong_shapes':RuntimeError,'corrupt_checkpoint':pickle.UnpicklingError}
    with pytest.raises(expected[kind]): inference.predict(image,cp,cfg)


@pytest.mark.parametrize('split',['train','invalid','',None])
def test_invalid_split_before_loading(split,saved,monkeypatch):
    cfg,_,cp=saved
    load=Mock(side_effect=AssertionError('Should reject first'))
    monkeypatch.setattr(inference,'load_model',load)
    with pytest.raises(ValueError,match='Split'): inference.evaluate_checkpoint(cp,cfg,split)
    load.assert_not_called()


def test_evaluate_checkpoint_uses_only_val_and_preprocessing(saved,monkeypatch):
    cfg,_,cp=saved
    sentinel=object(); transform=object()
    monkeypatch.setattr(inference,'build_preprocessing',Mock(return_value=transform))
    builder=Mock(return_value={'val':sentinel})
    monkeypatch.setattr(inference,'build_loaders',builder)
    evaluate=Mock(return_value={'loss':.5}); monkeypatch.setattr(inference,'evaluate',evaluate)
    assert inference.evaluate_checkpoint(cp,cfg)=={'loss':.5}
    assert builder.call_args.kwargs==dict(train_transform=transform,eval_transform=transform)
    assert evaluate.call_args.args[1] is sentinel
    assert not evaluate.call_args.args[0].training


def test_end_to_end_real_train_val_subset_restore_prediction_and_figures(tmp_path,monkeypatch,request):
    # Test-only small config: original CONFIG and architecture defaults remain unchanged.
    cfg=replace(CONFIG,root=tmp_path,image_size=(8,8),hidden_dims=[8,4],batch_size=3,epochs=2,device='cpu')
    loaders={}
    datasets={}
    for split,per_class in [('train',4),('val',3)]:
        transform=build_augmentation(cfg) if split=='train' else build_preprocessing(cfg.image_size)
        ds=HorseHumanDataset(CONFIG.splits_dir/f'{split}.csv',transform=transform)
        indices=[i for label in (0,1) for i in ds.samples.index[ds.samples.label==label][:per_class]]
        datasets[split]=(ds,indices)
        loaders[split]=DataLoader(Subset(ds,indices),batch_size=3,shuffle=False)
    model=MLP(cfg.input_dim,cfg.hidden_dims,cfg.activation)
    initialize_model(model)
    cp=tmp_path/'checkpoints'/'smoke.pt'; hp=tmp_path/'history'/'smoke.json'
    history=fit(model,loaders['train'],loaders['val'],torch.optim.Adam(model.parameters(),lr=.001),
                nn.BCEWithLogitsLoss(),'cpu',2,cp,hp)
    checkpoint=torch.load(cp,weights_only=True,map_location='cpu')
    # No Test loader is created or consumed by the smoke run.
    builder=Mock(return_value={'val':loaders['val']})
    monkeypatch.setattr(inference,'build_loaders',builder)
    stats=inference.evaluate_checkpoint(cp,cfg,split='val')
    assert stats['loss']==pytest.approx(history['best_val_loss'],abs=1e-7)
    assert checkpoint['epoch']==history['best_epoch']
    assert sum(stats[k] for k in ('tp','tn','fp','fn'))==6
    restored=inference.load_model(cp,cfg)
    # Independent per-example stable BCE formula confirms restoration loss.
    logits=[]; targets=[]
    with torch.no_grad():
        for x,y in loaders['val']:
            logits.extend(restored(x).flatten().tolist()); targets.extend(y.flatten().tolist())
    expected=sum(max(z,0)-z*y+math.log1p(math.exp(-abs(z))) for z,y in zip(logits,targets))/len(targets)
    assert stats['loss']==pytest.approx(expected,abs=1e-7)
    ds,indices=datasets['val']; image=CONFIG.root/ds.samples.iloc[indices[0]].image_path
    prediction=inference.predict(image,cp,cfg)
    assert prediction['human_probability']==pytest.approx(1/(1+math.exp(-logits[0])),abs=1e-7)
    plot_history(history,tmp_path/'figures'/'history.png')
    plot_confusion_matrix(stats,tmp_path/'figures'/'cm.png')
    assert all(p.stat().st_size>100 for p in (tmp_path/'figures').glob('*.png'))
    request.node.user_properties.append(('smoke',json.dumps(dict(train_samples=8,val_samples=6,epochs=2,
        best_epoch=history['best_epoch'],best_val_loss=history['best_val_loss'],restored_val_loss=stats['loss'],prediction=prediction))))
