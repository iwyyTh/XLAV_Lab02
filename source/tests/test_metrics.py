"""M6 metric oracles: explicit cases and sklearn, never production self-oracles."""
import pytest
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from source.metrics import binary_metrics


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('targets,probs,threshold',[
    ([0,1,0,1],[0.1,.9,.2,.8],.5),
    ([0,1,0,1],[.9,.1,.8,.2],.5),
    ([0,1,0,1],[1,1,1,1],.5),
    ([0,1,0,1],[0,0,0,0],.5),
    ([0,0],[0,0],.5),([1,1],[1,1],.5),
    ([0,1],[.49,.5],.5),([0,1,1,0],[.1,.75,.9,.8],.75),
    ([0,1],[0,1],0),([0,1],[0,1],1),
])
def test_metrics_match_independent_sklearn_and_preserve_inputs(dtype,targets,probs,threshold):
    y,p=torch.tensor(targets,dtype=dtype),torch.tensor(probs,dtype=dtype)
    old_y,old_p=y.clone(),p.clone()
    result=binary_metrics(y,p,threshold)
    predictions=[int(v>=threshold) for v in probs]
    expected=dict(accuracy=accuracy_score(targets,predictions),
                  precision=precision_score(targets,predictions,zero_division=0),
                  recall=recall_score(targets,predictions,zero_division=0),
                  f1=f1_score(targets,predictions,zero_division=0))
    tn,fp,fn,tp=confusion_matrix(targets,predictions,labels=[0,1]).ravel()
    expected.update(tn=tn,fp=fp,fn=fn,tp=tp)
    assert result==pytest.approx(expected)
    assert all(0<=result[k]<=1 for k in ('accuracy','precision','recall','f1'))
    assert torch.equal(y,old_y) and torch.equal(p,old_p)


@pytest.mark.parametrize('y,p',[
    ([0,1],[.5]), ([[0],[1]],[[.1],[.9]]),([],[]),
    ([2],[.5]),([-1],[.5]),([.5],[.5]),([float('nan')],[.5]),([float('inf')],[.5]),
    ([1],[-.01]),([1],[1.01]),([1],[float('nan')]),([1],[float('inf')]),([1],[float('-inf')])])
def test_invalid_tensor_values_and_shapes(y,p):
    with pytest.raises(ValueError):
        binary_metrics(torch.tensor(y),torch.tensor(p))


@pytest.mark.parametrize('threshold',[-.01,1.01,float('nan'),float('inf'),float('-inf')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        binary_metrics(torch.tensor([0,1]),torch.tensor([.1,.9]),threshold)
