"""Inspect plotted data, not just existence of exported images."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from source.visualization import plot_history,plot_confusion_matrix


@pytest.fixture
def figures(monkeypatch):
    captured=[]; original=plt.subplots
    def capture(*args,**kwargs):
        fig,axes=original(*args,**kwargs); captured.append(fig); return fig,axes
    monkeypatch.setattr(plt,'subplots',capture)
    yield captured
    for fig in captured: plt.close(fig)


def check_png(path):
    assert path.stat().st_size>100
    with Image.open(path) as image:
        image.verify()


def test_history_curves_marker_directory_and_close(tmp_path,figures):
    keys=['loss','accuracy','precision','recall','f1']
    history={'train':[dict(epoch=e,**{k:e*.1 for k in keys}) for e in (1,2,3)],
             'val':[dict(epoch=e,**{k:e*.2 for k in keys}) for e in (1,2,3)],'best_epoch':2}
    path=tmp_path/'nested'/'history.png'
    plot_history(history,path)
    check_png(path)
    fig=figures[-1]
    assert not plt.fignum_exists(fig.number)
    for ax in fig.axes[:5]:
        train,val,marker=ax.lines
        np.testing.assert_array_equal(train.get_xdata(),[1,2,3])
        np.testing.assert_allclose(train.get_ydata(),[.1,.2,.3])
        np.testing.assert_allclose(val.get_ydata(),[.2,.4,.6])
        np.testing.assert_array_equal(marker.get_xdata(),[2,2])
        assert train.get_label()=='Train' and val.get_label()=='Validation'


def test_confusion_orientation_counts_labels_directory_and_close(tmp_path,figures):
    path=tmp_path/'nested'/'cm.png'
    plot_confusion_matrix(dict(tn=7,fp=2,fn=3,tp=11),path)
    check_png(path)
    fig=figures[-1]; ax=fig.axes[0]
    np.testing.assert_array_equal(ax.images[0].get_array(),[[7,2],[3,11]])
    assert [t.get_text() for t in ax.get_xticklabels()]==['Horse','Human']
    assert [t.get_text() for t in ax.get_yticklabels()]==['Horse','Human']
    assert ax.get_xlabel()=='Predicted Label' and ax.get_ylabel()=='True Label'
    assert [t.get_text() for t in ax.texts]==['7','2','3','11']
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize('function',[plot_history,plot_confusion_matrix])
def test_missing_required_keys(function,tmp_path,figures):
    with pytest.raises(KeyError): function({},tmp_path/'bad.png')
    assert not (tmp_path/'bad.png').exists()
