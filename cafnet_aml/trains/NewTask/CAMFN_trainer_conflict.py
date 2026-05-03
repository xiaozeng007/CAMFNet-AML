
import torch
import torch.nn.functional as F

class ConflictLossMixin:
    def _bin(self, y):
        if y.dim() > 1: y = y.view(-1)
        return (y > 0).float()
    def _zero_tensor(self, *refs, device=None):
        for r in refs:
            if isinstance(r, torch.Tensor):
                return torch.zeros([], device=r.device)
        if device is not None:
            return torch.zeros([], device=device)
        return torch.zeros([])
    def loss_conflict(self, outputs, labels, weight=0.3):
        if not isinstance(outputs, dict) or ('tension_score' not in outputs):
            return torch.zeros([], device=labels.device)
        y = self._bin(labels)
        target = y - 0.5
        w = 1.0 + y  # sarcastic 2x
        return weight * ((w * (outputs['tension_score'] - target) ** 2).mean())
    def loss_divergence(self, outputs, labels=None, weight=0.1):
        if not isinstance(outputs, dict):
            return self._zero_tensor(device=labels.device if isinstance(labels, torch.Tensor) else None)
        l = outputs.get('feature_literal', None)
        i = outputs.get('feature_intended', None)
        if l is None or i is None:
            return self._zero_tensor(l, i, device=labels.device if isinstance(labels, torch.Tensor) else None)
        cos = F.cosine_similarity(l, i, dim=-1).mean()
        return weight * (1 - cos)
    def loss_gate(self, outputs, labels=None, weight=0.05):
        if not isinstance(outputs, dict) or ('modality_gate_values' not in outputs):
            return self._zero_tensor(device=labels.device if isinstance(labels, torch.Tensor) else None)
        g = outputs['modality_gate_values']
        if g.dim() != 1 or g.numel() != 3:
            g = g.mean(dim=0)
        g_text, g_audio, g_vision = g[0], g[1], g[2]
        return weight * ((1.0 - g_text)**2 + g_audio**2 + g_vision**2)
    def loss_contrast(self, outputs, weight=0.1):
        if not isinstance(outputs, dict):
            return self._zero_tensor()
        l = outputs.get('feature_literal', None)
        i = outputs.get('feature_intended', None)
        if l is None or i is None:
            return self._zero_tensor(l, i)
        if l.dim() != 2 or i.dim() != 2:
            return self._zero_tensor(l)
        sim = F.cosine_similarity(l, i, dim=-1)
        return weight * (1 - sim.mean())

def add_conflict_losses_to_total(L_main, outputs, labels, mixin: ConflictLossMixin,
                                 epoch, w_conflict=0.3, w_div=0.1, w_gate=0.05):
    epoch = max(1, epoch)
    factor = min(1.0, epoch / 5.0)
    L = L_main
    L = L + mixin.loss_conflict(outputs, labels, weight=w_conflict * factor)
    L = L + mixin.loss_divergence(outputs, labels, weight=w_div * factor)
    L = L + mixin.loss_gate(outputs, labels, weight=w_gate * (0.5 + factor))
    L = L + mixin.loss_contrast(outputs, weight=0.1)
    return L
