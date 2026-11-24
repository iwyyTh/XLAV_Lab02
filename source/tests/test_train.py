"""CLI orchestration without real training or consuming a Test loader."""
import json
import random
import sys
from dataclasses import replace,asdict
from unittest.mock import Mock
import numpy as np
import pytest
import torch
import source.train as cli


@pytest.fixture(autouse=True)
def preserve_rng():
    py,npstate=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(): yield
    random.setstate(py); np.random.set_state(npstate)


def test_parser_defaults(monkeypatch):
    monkeypatch.setattr(sys,'argv',['train'])
    a=cli.parse_args()
    for field in ('activation','initialization','epochs','learning_rate','batch_size','seed','device'):
        assert getattr(a,field)==getattr(cli.CONFIG,field)
    assert a.run_name=='baseline' and not a.no_augmentation


@pytest.mark.parametrize('flag,value',[('--activation','tanh'),('--initialization','uniform'),('--device','invalid'),('--epochs','abc'),('--unknown','x')])
def test_parser_invalid_arguments(flag,value,monkeypatch):
    monkeypatch.setattr(sys,'argv',['train',flag,value])
    with pytest.raises(SystemExit) as exc: cli.parse_args()
    assert exc.value.code==2


@pytest.mark.parametrize('args',[['--epochs','0'],['--batch-size','-1'],['--learning-rate','0'],['--learning-rate','nan'],['--learning-rate','inf'],['--device','cuda']])
def test_main_rejects_invalid_settings_before_loaders(args,monkeypatch):
    monkeypatch.setattr(sys,'argv',['train']+args)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    loaders=Mock(side_effect=AssertionError('Must reject before reading dataset'))
    monkeypatch.setattr(cli,'build_loaders',loaders)
    with pytest.raises(ValueError): cli.main()
    loaders.assert_not_called()


@pytest.mark.parametrize('activation',['relu','sigmoid'])
@pytest.mark.parametrize('initialization',['normal','fan_in_normal','xavier_normal','he_normal'])
@pytest.mark.parametrize('no_aug',[False,True])
def test_main_orchestration_and_saved_config(activation,initialization,no_aug,tmp_path,monkeypatch):
    cfg=replace(cli.CONFIG,root=tmp_path)
    monkeypatch.setattr(cli,'CONFIG',cfg)
    original=asdict(cfg)
    argv=['train','--activation',activation,'--initialization',initialization,'--epochs','2',
          '--batch-size','7','--learning-rate','.002','--seed','19','--device','cpu','--run-name','unit']
    if no_aug: argv.append('--no-augmentation')
    monkeypatch.setattr(sys,'argv',argv)
    pre,aug=object(),object()
    monkeypatch.setattr(cli,'build_preprocessing',Mock(return_value=pre))
    monkeypatch.setattr(cli,'build_augmentation',Mock(return_value=aug))
    train,val=object(),object()
    class Loaders(dict):
        def __getitem__(self,key):
            assert key!='test', 'Test must not be selected during training'
            return super().__getitem__(key)
    loaders=Mock(return_value=Loaders(train=train,val=val))
    monkeypatch.setattr(cli,'build_loaders',loaders)
    # Small registered model isolates CLI wiring; real MLP is covered in smoke.
    factory=Mock(return_value=torch.nn.Linear(1,1))
    monkeypatch.setattr(cli,'MLP',factory)
    init=Mock(); monkeypatch.setattr(cli,'initialize_model',init)
    fit=Mock(return_value={'best_epoch':1,'best_val_loss':.3})
    monkeypatch.setattr(cli,'fit',fit)
    seed=Mock(wraps=cli.set_seed); monkeypatch.setattr(cli,'set_seed',seed)
    cli.main()
    actual=loaders.call_args.args[0]
    assert actual.batch_size==7 and actual.learning_rate==.002 and actual.seed==19
    assert loaders.call_args.args[1:] == (pre if no_aug else aug,pre)
    assert cli.build_augmentation.call_count==int(not no_aug)
    factory.assert_called_once_with(cfg.input_dim,cfg.hidden_dims,activation)
    init.assert_called_once_with(factory.return_value,initialization)
    seed.assert_called_once_with(19)
    a=fit.call_args.args
    assert a[1] is train and a[2] is val
    assert isinstance(a[3],torch.optim.Adam) and a[3].param_groups[0]['lr']==.002
    assert isinstance(a[4],torch.nn.BCEWithLogitsLoss) and a[5:7]==('cpu',2)
    assert a[7]==tmp_path/'results/checkpoints/unit.pt'
    assert a[8]==tmp_path/'results/histories/unit.json'
    saved=json.loads((tmp_path/'results/configs/unit.json').read_text())
    assert saved=={**asdict(actual),'root':str(tmp_path),'image_size':list(actual.image_size),
                   'run_name':'unit','augmentation_enabled':not no_aug}
    assert asdict(cfg)==original


def test_set_seed_all_three_rngs():
    def draw(): return random.random(),np.random.rand(),torch.rand(3)
    cli.set_seed(42); a=draw()
    cli.set_seed(42); b=draw()
    cli.set_seed(43); c=draw()
    assert a[0]==b[0] and a[1]==b[1] and torch.equal(a[2],b[2])
    assert a[0]!=c[0] and a[1]!=c[1] and not torch.equal(a[2],c[2])
