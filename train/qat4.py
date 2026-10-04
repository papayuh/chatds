"""Experimental project-owned 4-bit weight fake-quant (STE).
No DSQ4 exporter or device support is shipped. enable(True) makes every nn.Linear and
nn.Embedding weight go through it in forward; the tied token/classifier weight
is quantized the same way the exporter does (rows=vocab, groups along dim)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

GS = 32
ON = False


def _round_half_away(x):  # C roundf_i(): trunc(v +- 0.5)
    return torch.trunc(x + torch.where(x >= 0, torch.full_like(x, 0.5), torch.full_like(x, -0.5)))


class WQAT4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, gs=GS):
        out_dim, in_dim = w.shape
        if in_dim % gs:
            return w
        wf = w.float()
        wg = wf.view(out_dim, in_dim // gs, gs)
        rmax = wf.abs().amax(dim=-1, keepdim=True).view(out_dim, 1, 1)
        gmax = wg.abs().amax(dim=-1, keepdim=True)
        rsc = rmax / 7.0
        s16 = torch.trunc(torch.where(rmax > 0, gmax / rmax.clamp_min(1e-12) * 32767.0,
                                      torch.zeros_like(gmax)) + 0.5)
        eff = rsc * s16 / 32768.0
        inv = torch.where(eff > 0, 1.0 / eff.clamp_min(1e-30), torch.zeros_like(eff))
        q = _round_half_away(wg * inv).clamp(-8, 7)
        return (q * eff).view(out_dim, in_dim).type_as(w)

    @staticmethod
    def backward(ctx, grad):
        return grad, None  # straight-through


def _wq(w):
    return WQAT4.apply(w, GS) if ON else w


_installed = False


def enable(on=True, gs=32):
    """Idempotent. Patches Linear/Embedding.forward once; ON toggles the quant."""
    global ON, GS, _installed
    ON, GS = bool(on), gs
    if not _installed:
        nn.Linear.forward = lambda self, x: F.linear(x, _wq(self.weight), self.bias)
        nn.Embedding.forward = lambda self, x: F.embedding(x, _wq(self.weight))
        _installed = True
