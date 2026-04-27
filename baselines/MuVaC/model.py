import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import BertTextEncoder

__all__ = ["MuVaC"]


class MuVaC(nn.Module):
    """
    MuVaC full training model with:
    - Variational bottleneck (mu/logvar + KL)
    - Cross-modal gated fusion
    - Modality reconstruction heads
    - Contrastive projection heads
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.use_bert = getattr(args, "use_bert", True)
        dropout = getattr(args, "dropout", 0.3)
        proj_dim = getattr(args, "proj_dim", 256)
        latent_dim = getattr(args, "latent_dim", 128)
        fusion_dim = getattr(args, "fusion_dim", 256)
        ctr_dim = getattr(args, "contrast_dim", 128)

        t_dim, a_dim, v_dim = args.feature_dims
        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=getattr(args, "use_finetune", True),
                transformers=getattr(args, "transformers", "bert"),
                pretrained=getattr(args, "pretrained", "bert-base-uncased"),
            )
            t_dim = 768
        else:
            self.text_model = None

        self.t_proj = nn.Sequential(
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.a_proj = nn.Sequential(
            nn.LayerNorm(a_dim),
            nn.Linear(a_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.v_proj = nn.Sequential(
            nn.LayerNorm(v_dim),
            nn.Linear(v_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim * 3),
            nn.Sigmoid(),
        )

        self.mu = nn.Linear(proj_dim * 3, latent_dim)
        self.logvar = nn.Linear(proj_dim * 3, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_dim, self.output_dim)

        # Reconstruction heads for modality-aware VAE training.
        self.rec_t = nn.Linear(latent_dim, proj_dim)
        self.rec_a = nn.Linear(latent_dim, proj_dim)
        self.rec_v = nn.Linear(latent_dim, proj_dim)

        # Contrastive heads.
        self.ctr_t = nn.Linear(proj_dim, ctr_dim)
        self.ctr_a = nn.Linear(proj_dim, ctr_dim)
        self.ctr_v = nn.Linear(proj_dim, ctr_dim)
        self.ctr_z = nn.Linear(latent_dim, ctr_dim)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not mu.requires_grad:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _pool_seq(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x[:, 0, :] if x.dim() == 3 else x

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)  # [B, L, 768]

        t_feat = self._pool_seq(self.t_proj(text))
        a_feat = self.a_proj(audio).mean(dim=1)
        v_feat = self.v_proj(video).mean(dim=1)

        concat = torch.cat([t_feat, a_feat, v_feat], dim=-1)
        gates = self.cross_gate(concat).view(concat.size(0), 3, -1)
        stacked = torch.stack([t_feat, a_feat, v_feat], dim=1) * gates
        fused = stacked.reshape(stacked.size(0), -1)

        mu = self.mu(fused)
        logvar = self.logvar(fused)
        z = self._reparameterize(mu, logvar)
        feat_f = self.decoder(z)
        out = self.head(feat_f)

        rec_t = self.rec_t(z)
        rec_a = self.rec_a(z)
        rec_v = self.rec_v(z)

        z_t = F.normalize(self.ctr_t(t_feat), dim=-1)
        z_a = F.normalize(self.ctr_a(a_feat), dim=-1)
        z_v = F.normalize(self.ctr_v(v_feat), dim=-1)
        z_lat = F.normalize(self.ctr_z(z), dim=-1)

        kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=-1))
        rec = (F.mse_loss(rec_t, t_feat) + F.mse_loss(rec_a, a_feat) + F.mse_loss(rec_v, v_feat)) / 3.0

        return {
            "Feature_t": t_feat,
            "Feature_a": a_feat,
            "Feature_v": v_feat,
            "Feature_f": feat_f,
            "M": out,
            "mu": mu,
            "logvar": logvar,
            "z_t": z_t,
            "z_a": z_a,
            "z_v": z_v,
            "z_lat": z_lat,
            "rec_t": rec_t,
            "rec_a": rec_a,
            "rec_v": rec_v,
            "KL": kl,
            "REC": rec,
        }

