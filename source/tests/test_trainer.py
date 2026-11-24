"""M6 epoch semantics and checkpoint selection using controlled tiny datasets."""
import json
import math
from unittest.mock import Mock
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import source.trainer as trainer
import source.activations as acts
from source.model import MLP


@pytest.fixture(autouse=True)
def rng():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        yield


@pytest.fixture(params=['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.fixture
def data():
    # Batches of 2,2,1: global F1=.4 vs batch-F1 average=1/3.
    x=torch.tensor([[-2.],[2.],[2.],[-2.],[2.]])
    y=torch.tensor([[0.],[1.],[0.],[1.],[0.]])
    return DataLoader(TensorDataset(x,y),batch_size=2),x,y


@pytest.mark.parametrize('training',[False,True])
def test_weighted_loss_global_metrics_modes_detach_and_zero_grad(training,data,device,monkeypatch):
    loader,x,y=data
    model=nn.Linear(1,1).to(device)
    with torch.no_grad():
        model.weight.fill_(1); model.bias.zero_()
    model.train(not training)
    snapshots=[]
    def observe(module,args,output):
        snapshots.append((module.training,torch.is_grad_enabled(),args[0].device.type))
    handle=model.register_forward_hook(observe)
    optimizer=torch.optim.SGD(model.parameters(),lr=0)
    zero=Mock(wraps=optimizer.zero_grad); step=Mock(wraps=optimizer.step)
    monkeypatch.setattr(optimizer,'zero_grad',zero); monkeypatch.setattr(optimizer,'step',step)
    original=trainer.binary_metrics
    def checked(y,p):
        assert not y.requires_grad and not p.requires_grad and p.grad_fn is None
        assert y.device.type==p.device.type=='cpu' and len(y)==5
        return original(y,p)
    monkeypatch.setattr(trainer,'binary_metrics',checked)
    before=[p.clone() for p in model.parameters()]
    if training:
        for p in model.parameters(): p.grad=torch.full_like(p,999)
        result=trainer.train_epoch(model,loader,optimizer,nn.BCEWithLogitsLoss(),device)
        assert zero.call_count==step.call_count==3
        # Last sample x=2,y=0: exact BCE gradient; detects stale accumulated grads.
        assert model.weight.grad.item()==pytest.approx(2/(1+math.exp(-2)),rel=1e-6)
    else:
        monkeypatch.setattr(torch.Tensor,'backward',Mock(side_effect=AssertionError('No backward in eval')))
        result=trainer.evaluate(model,loader,nn.BCEWithLogitsLoss(),device)
        assert trainer.evaluate(model,loader,nn.BCEWithLogitsLoss(),device)==result
        assert zero.call_count==step.call_count==0
        assert all(p.grad is None for p in model.parameters())
    handle.remove()
    expected_loss=sum(max(z,0)-z*t+math.log1p(math.exp(-abs(z))) for z,t in zip(x.flatten().tolist(),y.flatten().tolist()))/5
    assert result==pytest.approx(dict(loss=expected_loss,accuracy=.4,precision=1/3,recall=.5,f1=.4,tp=1,tn=1,fp=2,fn=1))
    assert all(mode==training and grad==training and dev==device for mode,grad,dev in snapshots)
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    assert all(isinstance(v,(float,int)) for v in result.values())


def test_training_custom_backward_updates_parameters(data,monkeypatch):
    loader,_,_=data
    model=MLP(input_dim=1,hidden_dims=(4,),activation='sigmoid')
    spy=Mock(wraps=acts.activation_derivative)
    monkeypatch.setattr(acts,'activation_derivative',spy)
    old=[p.clone() for p in model.parameters()]
    result=trainer.train_epoch(model,loader,torch.optim.Adam(model.parameters()),nn.BCEWithLogitsLoss(),'cpu')
    assert spy.call_count==3 and math.isfinite(result['loss'])
    assert all(not torch.equal(a,b) for a,b in zip(old,model.parameters()))


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('bad',[None,float('nan'),float('inf')])
def test_empty_loader_and_nonfinite_loss(training,bad,data):
    loader,_,_=data
    model=nn.Linear(1,1)
    opt=torch.optim.SGD(model.parameters(),lr=.1)
    if bad is None:
        loader=DataLoader(TensorDataset(torch.empty(0,1),torch.empty(0,1)),batch_size=2)
        criterion=nn.BCEWithLogitsLoss(); error=ValueError
    else:
        criterion=lambda logits,targets: logits.sum()*0+bad
        error=FloatingPointError
    before=[p.clone() for p in model.parameters()]
    with pytest.raises(error):
        if training: trainer.train_epoch(model,loader,opt,criterion,'cpu')
        else: trainer.evaluate(model,loader,criterion,'cpu')
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))


def test_fit_order_best_checkpoint_ties_history_and_directories(tmp_path,monkeypatch):
    model=nn.Linear(1,1); opt=torch.optim.Adam(model.parameters())
    events=[]; epoch=0; losses=[.8,.3,.3,.5]
    def train(*args):
        nonlocal epoch
        epoch+=1; events.append(('train',epoch))
        with torch.no_grad(): model.weight.fill_(epoch)
        return dict(loss=epoch,f1=.1)
    def evaluate(*args):
        events.append(('val',epoch)); return dict(loss=losses[epoch-1],f1=.2)
    monkeypatch.setattr(trainer,'train_epoch',train); monkeypatch.setattr(trainer,'evaluate',evaluate)
    save=Mock(wraps=torch.save); monkeypatch.setattr(torch,'save',save)
    cp=tmp_path/'ck'/'best.pt'; hp=tmp_path/'hist'/'run.json'
    history=trainer.fit(model,object(),object(),opt,nn.BCEWithLogitsLoss(),'cpu',4,cp,hp)
    assert events==[(kind,e) for e in range(1,5) for kind in ('train','val')]
    assert save.call_count==2 # Strictly lower only; ties preserve earliest best.
    ck=torch.load(cp,weights_only=True,map_location='cpu')
    assert set(ck)=={'epoch','model_state_dict','optimizer_state_dict','best_val_loss'}
    assert ck['epoch']==history['best_epoch']==2
    assert ck['best_val_loss']==history['best_val_loss']==.3
    assert ck['model_state_dict']['weight'].item()==2
    assert len(history['train'])==len(history['val'])==4
    assert [s['loss'] for s in history['val']]==losses
    assert [s['epoch'] for s in history['train']]==[1,2,3,4]
    assert json.loads(hp.read_text())==history


@pytest.mark.parametrize('epochs',[0,-1,True,1.5,'2',None])
def test_fit_invalid_epochs(epochs,tmp_path):
    with pytest.raises(ValueError,match='positive integer'):
        trainer.fit(None,None,None,None,None,'cpu',epochs,tmp_path/'x.pt',tmp_path/'x.json')
    assert not list(tmp_path.iterdir())
