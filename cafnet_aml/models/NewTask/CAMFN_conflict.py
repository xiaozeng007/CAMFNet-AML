
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityGates(nn.Module):
    def __init__(self, init=(0.8, 0.1, 0.1), temperature=2.0, noise_std=0.05):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(list(init), dtype=torch.float32))
        self.temperature = temperature
        self.noise_std = noise_std
    def forward(self):
        base = F.softmax(self.logits * self.temperature, dim=0)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(base) * self.noise_std
            base = F.softmax(base + noise, dim=0)
        return base

class StableTensionRegressor(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

class ConflictHeads(nn.Module):
    def __init__(self, feat_dim, proj_dim):
        super().__init__()
        self.proj_l = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Tanh(),
            nn.Dropout(0.4)
        )
        self.proj_i = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.tension = StableTensionRegressor(in_dim=proj_dim + 1, hidden=proj_dim)
    def forward(self, lit_feat, int_feat, lit_prob=None, int_prob=None):
        l = self.proj_l(lit_feat)
        i = self.proj_i(int_feat)
        diff = torch.abs(l - i)
        if lit_prob is not None and int_prob is not None:
            gap = lit_prob - int_prob
            pgap = gap.unsqueeze(-1)
            inv = torch.sigmoid(gap * 5)
        else:
            pgap = torch.zeros(l.size(0), 1, device=l.device)
            inv = torch.ones(l.size(0), device=l.device)
        tens_input = torch.cat([diff, pgap], dim=-1)
        tens = 0.3 * torch.tanh(self.tension(tens_input)).squeeze(-1)
        tension_score = tens * (1 + inv)
        return {'feature_literal': l, 'feature_intended': i, 'tension_score': tension_score}

def wrap_camfn_with_conflict(model: nn.Module,
                             feat_dim: int,
                             proj_dim: int = None,
                             gate_init=(0.8, 0.1, 0.1),
                             gate_temperature: float = 2.0) -> nn.Module:
    if proj_dim is None:
        proj_dim = max(4, feat_dim // 2)
    model.gates = ModalityGates(init=gate_init, temperature=gate_temperature)
    model.conflict = ConflictHeads(feat_dim=feat_dim, proj_dim=proj_dim)
    orig_forward = model.forward
    def forward_with_conflict(*args, **kwargs):
        res = orig_forward(*args, **kwargs)
        if isinstance(res, dict) and ('Feature_t' in res) and ('Feature_f' in res) and ('T' in res) and ('M' in res):
            text_feat = res['Feature_t']
            fuse_feat = res['Feature_f']
            lit_prob = torch.softmax(res['T'], dim=-1)[:, 1]
            int_prob = torch.softmax(res['M'], dim=-1)[:, 1]
            conf = model.conflict(text_feat, fuse_feat, lit_prob=lit_prob, int_prob=int_prob)
            res.update(conf)
        g = model.gates()
        res['modality_gate_values'] = g.detach()
        return res
    model.forward = types.MethodType(forward_with_conflict, model)
    return model
