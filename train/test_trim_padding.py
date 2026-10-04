"""Independent padding trim must preserve all supervised logits/gradients."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_instruct as ti


def dataset(tmp_path, trim=True, packing='independent'):
    ids=np.zeros((2,32),dtype=np.uint16)
    tgt=np.full((2,32),-1,dtype=np.int32)
    ids[0,:5]=[1,4,5,6,2];tgt[0,2:4]=[6,2]
    ids[1,:7]=[1,7,8,9,10,11,2];tgt[1,2:6]=[9,10,11,2]
    ids.tofile(tmp_path/'train.ids.bin');tgt.tofile(tmp_path/'train.tgt.bin')
    return ti.MaskedShardDataset(str(tmp_path),'train',32,16,
        {'packing':packing,'seq_len':32,'vocab_size':16},trim_padding=trim)


class BothRows:
    def integers(self,*args,**kwargs):
        return np.array([0,1])


def test_trim_preserves_last_supervised_eos_and_prefixes(tmp_path,monkeypatch):
    monkeypatch.setattr(ti,'device_type','cpu')
    ds=dataset(tmp_path)
    x,y=ds.get_batch(2,'cpu',BothRows())
    assert x.shape == y.shape == (2,6)
    assert y[1,-1] == 2
    assert torch.equal(x,torch.tensor(ds.ids[:,:6].astype(np.int64)))
    assert torch.equal(y,torch.tensor(ds.tgt[:,:6].astype(np.int64)))
    assert x.is_contiguous() and y.is_contiguous()


def test_default_can_keep_full_window(tmp_path,monkeypatch):
    monkeypatch.setattr(ti,'device_type','cpu')
    x,y=dataset(tmp_path,trim=False).get_batch(2,'cpu',BothRows())
    assert x.shape == y.shape == (2,32)


def test_stream_cannot_masquerade_as_independent_padding(tmp_path):
    with pytest.raises(SystemExit,match='independent'):
        dataset(tmp_path,packing='stream')


def test_supervised_loss_and_gradients_match_untrimmed(tmp_path,monkeypatch):
    monkeypatch.setattr(ti,'device_type','cpu')
    ds=dataset(tmp_path)
    short=ds.get_batch(2,'cpu',BothRows())
    full=(torch.tensor(ds.ids.astype(np.int64)),torch.tensor(ds.tgt.astype(np.int64)))
    torch.manual_seed(123)
    model=ti.Transformer(ti.ModelArgs(dim=32,n_layers=2,n_heads=2,n_kv_heads=2,
        vocab_size=16,multiple_of=32,max_seq_len=32,dropout=0.0))
    model(*full);lf=model.last_loss;lf.backward()
    gradients=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    model(*short);ls=model.last_loss;ls.backward()
    torch.testing.assert_close(lf,ls,rtol=1e-5,atol=1e-6)
    for old,p in zip(gradients,model.parameters()):
        torch.testing.assert_close(old,p.grad,rtol=1e-5,atol=1e-6)
