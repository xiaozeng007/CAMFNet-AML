"""

Text-enhanced fusion network with explicit conflict modeling for MUStARD++.

Implements TET-style cross-modal refinement, distribution-level conflict scoring,

and shared multi-head predictions for sarcasm and emotion branches.

"""

import math

from typing import Dict, Optional, Tuple



import torch

import torch.nn as nn



from ..subNets import BertTextEncoder



__all__ = ['CAMFN']





def _build_mask_from_lengths(lengths: Optional[torch.Tensor], max_len: int, device: torch.device) -> Optional[torch.Tensor]:

    if lengths is None:

        return None

    if not torch.is_tensor(lengths):

        lengths = torch.tensor(lengths, device=device)

    lengths = lengths.to(device).long()

    seq_range = torch.arange(max_len, device=device).unsqueeze(0)

    return seq_range >= lengths.unsqueeze(1)


def _parse_feature_names(value):

    if value is None:

        return None

    if isinstance(value, str):

        return [x.strip() for x in value.split(',') if x.strip()]

    if isinstance(value, (list, tuple, set)):

        return [str(x).strip() for x in value if str(x).strip()]

    one = str(value).strip()

    return [one] if one else []





class TemporalProjector(nn.Module):

    """

    Bidirectional LSTM + temporal conv projector that preserves the time dimension.

    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, d_model: int = 128,

                 num_layers: int = 1, dropout: float = 0.1, kernel_size: int = 3):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, padding=padding)

        lstm_dropout = 0.0 if num_layers == 1 else dropout

        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,

                            dropout=lstm_dropout, batch_first=True, bidirectional=True)

        self.proj = nn.Sequential(

            nn.LayerNorm(hidden_dim * 2),

            nn.Linear(hidden_dim * 2, d_model),

            nn.GELU(),

            nn.Dropout(dropout)

        )



    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if x.dim() == 2:

            x = x.unsqueeze(0)

        x = x.float()

        conv_in = x.transpose(1, 2)

        conv_out = self.conv(conv_in).transpose(1, 2)

        outputs, _ = self.lstm(conv_out)

        proj = self.proj(outputs)

        mask = _build_mask_from_lengths(lengths, proj.size(1), proj.device)

        return proj, mask





class TETLayer(nn.Module):

    """

    Cross-attention layer (target <- context) with residual FFN.

    """

    def __init__(self, d_model: int, nheads: int = 4, attn_dropout: float = 0.1, ff_dropout: float = 0.1):

        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, nheads, dropout=attn_dropout, batch_first=True)

        self.ff = nn.Sequential(

            nn.Linear(d_model, d_model * 4),

            nn.GELU(),

            nn.Dropout(ff_dropout),

            nn.Linear(d_model * 4, d_model),

            nn.Dropout(ff_dropout)

        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)



    def forward(self, target: torch.Tensor, context: torch.Tensor,

                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        attn_out, _ = self.cross_attn(target, context, context, key_padding_mask=context_mask)

        target = self.norm1(target + attn_out)

        ff_out = self.ff(target)

        target = self.norm2(target + ff_out)

        return target





class TextEnhancedBlock(nn.Module):

    """

    Stack of TET cross-attention layers.

    """

    def __init__(self, d_model: int, layers: int = 2, nheads: int = 4,

                 attn_dropout: float = 0.1, ff_dropout: float = 0.1):

        super().__init__()

        self.layers = nn.ModuleList([

            TETLayer(d_model, nheads, attn_dropout, ff_dropout) for _ in range(layers)

        ])



    def forward(self, target: torch.Tensor, context: torch.Tensor,

                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        x = target

        for layer in self.layers:

            x = layer(x, context, context_mask)

        return x





class TriModalEncoder(nn.Module):

    """

    Single-layer transformer encoder operating over concatenated modal tokens.

    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=num_heads,

            dim_feedforward=d_model * 4,

            dropout=dropout,

            batch_first=True,

            activation='gelu'

        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)



    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        return self.encoder(tokens, src_key_padding_mask=mask)





class ConflictDistribution(nn.Module):

    """

    Distribution-level conflict scorer with lightweight residual gating.

    Supports conflict-feature ablation by dropping/replacing feature dimensions.

    """

    def __init__(self, fusion_dim: int, num_emotions: int, temperature_exp: float = 1.0,

                 temperature_imp: float = 1.0, alpha_init: float = 0.5, dropout: float = 0.1,

                 weight_js: float = 1.0, weight_pol: float = 0.6, weight_hard: float = 0.5,

                 weight_entropy: float = 0.3, weight_margin: float = 0.2, bias: float = 0.0,

                 valence: Optional[torch.Tensor] = None, feature_keep=None, feature_drop=None,
                 feature_add=None, extra_weight_inits: Optional[Dict[str, float]] = None):

        super().__init__()

        self.temperature_exp = temperature_exp

        self.temperature_imp = temperature_imp

        self.bias = nn.Parameter(torch.tensor(float(bias)))

        self.alpha_param = nn.Parameter(torch.tensor(alpha_init).float())

        self.base_feature_order = ['js', 'polar_gap', 'hard', 'entropy_inv', 'margin_inv']

        self.extra_feature_order = ['l1', 'conf_gap', 'margin_gap']

        aliases = {
            'jsd': 'js',
            'pol': 'polar_gap',
            'polar': 'polar_gap',
            'entropy': 'entropy_inv',
            'margin': 'margin_inv',
            'peak_gap': 'conf_gap',
            'confidence_gap': 'conf_gap'
        }

        supported = set(self.base_feature_order + self.extra_feature_order)

        def _normalize(names):

            raw = _parse_feature_names(names)

            if raw is None:

                return None

            out = []

            for name in raw:

                key = aliases.get(name.lower(), name.lower())

                if key in supported and key not in out:

                    out.append(key)

            return out

        keep_names = _normalize(feature_keep)

        drop_names = set(_normalize(feature_drop) or [])

        add_names = _normalize(feature_add) or []

        if keep_names:

            active = keep_names

        else:

            active = [name for name in self.base_feature_order if name not in drop_names]

            for name in add_names:

                if name not in active:

                    active.append(name)

        if not active:

            active = list(self.base_feature_order)

        self.active_feature_names = tuple(active)

        weight_init = {
            'js': float(weight_js),
            'polar_gap': float(weight_pol),
            'hard': float(weight_hard),
            'entropy_inv': float(weight_entropy),
            'margin_inv': float(weight_margin),
            'l1': 0.2,
            'conf_gap': 0.2,
            'margin_gap': 0.2
        }

        if extra_weight_inits:

            for name, value in extra_weight_inits.items():

                if name in weight_init:

                    weight_init[name] = float(value)

        self.feature_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight_init[name]))
            for name in self.active_feature_names
        })

        self.adapter = nn.Sequential(

            nn.Linear(len(self.active_feature_names), fusion_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(fusion_dim, fusion_dim),

            nn.GELU()

        )

        self.ln = nn.LayerNorm(fusion_dim)

        if valence is None:

            valence = torch.linspace(-1.0, 1.0, steps=num_emotions)

        self.register_buffer('valence', valence.float())



    @staticmethod

    def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:

        m = 0.5 * (p + q)

        kl_pm = torch.clamp(p, 1e-8).log() - torch.clamp(m, 1e-8).log()

        kl_qm = torch.clamp(q, 1e-8).log() - torch.clamp(m, 1e-8).log()

        return 0.5 * (torch.sum(p * kl_pm, dim=-1) + torch.sum(q * kl_qm, dim=-1))



    @staticmethod

    def _entropy(p: torch.Tensor) -> torch.Tensor:

        return -torch.sum(torch.clamp(p, 1e-8) * torch.clamp(p, 1e-8).log(), dim=-1)



    @staticmethod

    def _margin(prob: torch.Tensor) -> torch.Tensor:

        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values

        if top2.size(-1) == 1:

            diff = top2[..., 0]

        else:

            diff = top2[..., 0] - top2[..., 1]

        return torch.clamp(diff, 0.0, 1.0)



    def forward(self, fusion_feat: torch.Tensor, explicit_logits: torch.Tensor,

                implicit_logits: torch.Tensor, teacher: Optional[Dict[str, torch.Tensor]] = None,

                enable: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        batch = fusion_feat.size(0)

        device = fusion_feat.device

        zeros = fusion_feat.new_zeros(batch)

        stats_placeholder = {

            'js': zeros,

            'polar_gap': zeros,

            'hard': zeros,

            'entropy': zeros,

            'margin': zeros,

            'l1': zeros,

            'conf_gap': zeros,

            'margin_gap': zeros,

            'active_names': self.active_feature_names

        }

        if not enable:

            return fusion_feat, zeros, stats_placeholder



        with torch.no_grad():

            exp_source = teacher['explicit'] if teacher and 'explicit' in teacher else explicit_logits

            imp_source = teacher['implicit'] if teacher and 'implicit' in teacher else implicit_logits

            p = torch.softmax(exp_source / self.temperature_exp, dim=-1)

            q = torch.softmax(imp_source / self.temperature_imp, dim=-1)

            js = self._js_divergence(p, q) / math.log(2)

            num_classes = p.size(-1)

            if self.valence is None or self.valence.numel() == 0:

                valence_vec = torch.linspace(-1.0, 1.0, steps=num_classes, device=device)

            else:

                valence_vec = self.valence.to(device)

                if valence_vec.size(0) != num_classes:

                    valence_vec = torch.linspace(-1.0, 1.0, steps=num_classes, device=device)

            pol_p = torch.sum(p * valence_vec, dim=-1)

            pol_q = torch.sum(q * valence_vec, dim=-1)

            pol_gap = torch.abs(pol_p - pol_q) / 2.0

            hard = (torch.argmax(p, dim=-1) != torch.argmax(q, dim=-1)).float()

            entropy = 0.5 * (self._entropy(p) + self._entropy(q)) / math.log(p.size(-1))

            margin = 0.5 * (self._margin(p) + self._margin(q))

            l1 = torch.mean(torch.abs(p - q), dim=-1)

            conf_gap = torch.abs(torch.max(p, dim=-1).values - torch.max(q, dim=-1).values)

            margin_gap = torch.abs(self._margin(p) - self._margin(q))

        feature_bank = {
            'js': js,
            'polar_gap': pol_gap,
            'hard': hard,
            'entropy_inv': 1 - entropy,
            'margin_inv': 1 - margin,
            'l1': l1,
            'conf_gap': conf_gap,
            'margin_gap': margin_gap
        }

        score = fusion_feat.new_zeros(batch)

        for name in self.active_feature_names:

            score = score + self.feature_weights[name] * feature_bank[name]

        score = score + self.bias

        conf = torch.sigmoid(score)

        adapter_input = torch.stack([feature_bank[name] for name in self.active_feature_names], dim=-1)

        adapted = self.adapter(adapter_input)

        alpha = torch.sigmoid(self.alpha_param)

        fused = self.ln(fusion_feat + alpha * conf.unsqueeze(-1) * (adapted - fusion_feat))

        stats = {

            'js': js,

            'polar_gap': pol_gap,

            'hard': hard,

            'entropy': entropy,

            'margin': margin,

            'l1': l1,

            'conf_gap': conf_gap,

            'margin_gap': margin_gap,

            'active_names': self.active_feature_names

        }

        return fused, conf, stats



class EmotionTriadFusion(nn.Module):

    """

    Fuse explicit/implicit emotion distributions with affect cues

    and inject them back to the multimodal backbone.

    """

    def __init__(self, num_emotions: int, fusion_dim: int, affect_dim: int = 2,

                 triad_dim: int = 128, heads: int = 4, dropout: float = 0.1):

        super().__init__()

        self.num_emotions = num_emotions

        self.affect_dim = affect_dim

        self.exp_proj = nn.Sequential(

            nn.LayerNorm(num_emotions),

            nn.Linear(num_emotions, triad_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

        self.imp_proj = nn.Sequential(

            nn.LayerNorm(num_emotions),

            nn.Linear(num_emotions, triad_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

        self.affect_proj = nn.Sequential(

            nn.LayerNorm(affect_dim),

            nn.Linear(affect_dim, triad_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

        self.query_proj = nn.Linear(fusion_dim, triad_dim)

        self.attn = nn.MultiheadAttention(embed_dim=triad_dim, num_heads=heads,

                                          dropout=dropout, batch_first=True)

        self.context_proj = nn.Linear(triad_dim, fusion_dim)

        self.summary_proj = nn.Sequential(

            nn.Linear(triad_dim * 3, fusion_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

        self.gate = nn.Sequential(

            nn.Linear(fusion_dim * 2, fusion_dim),

            nn.GELU(),

            nn.Linear(fusion_dim, fusion_dim),

            nn.Sigmoid()

        )

        self.out_norm = nn.LayerNorm(fusion_dim)



    def forward(self, fusion_feat: torch.Tensor, explicit_logits: torch.Tensor,

                implicit_logits: torch.Tensor, affect_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p_exp = torch.softmax(explicit_logits, dim=-1)

        p_imp = torch.softmax(implicit_logits, dim=-1)

        exp_embed = self.exp_proj(p_exp)

        imp_embed = self.imp_proj(p_imp)

        aff_embed = self.affect_proj(affect_vec)

        tokens = torch.stack([exp_embed, imp_embed, aff_embed], dim=1)

        query = self.query_proj(fusion_feat).unsqueeze(1)

        attn_out, attn_weights = self.attn(query, tokens, tokens)

        context = self.context_proj(attn_out.squeeze(1))

        gate = self.gate(torch.cat([fusion_feat, context], dim=-1))

        refined = self.out_norm(fusion_feat + gate * context)

        summary = self.summary_proj(torch.cat([exp_embed, imp_embed, aff_embed], dim=-1))

        return refined, attn_weights.squeeze(1), summary





class CAMFN(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        self.use_text = bool(getattr(args, 'use_text', True))
        self.use_audio = bool(getattr(args, 'use_audio', True))
        self.use_video = bool(getattr(args, 'use_video', True))

        self.aligned = args.need_data_aligned

        self.train_mode = getattr(args, 'train_mode', 'classification')

        self.is_classification = str(self.train_mode).lower() == 'classification'

        self.use_reg_head = getattr(args, 'use_reg_head', not self.is_classification)

        # dataset-specific switches (regression datasets can turn off heavy heads)

        self.enable_conflict = getattr(args, 'enable_conflict', self.is_classification)

        self.enable_triad = getattr(args, 'enable_triad', self.is_classification)

        self.num_emotions = getattr(args, 'num_emotions', 7)

        self.dropout = getattr(args, 'dropout', 0.3)

        self.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)

        self.d_model = getattr(args, 'fusion_proj_dim', getattr(args, 'proj_dim', 128))

        self.fusion_dim = getattr(args, 'fusion_dim', self.d_model)

        self.affect_dim = getattr(args, 'affect_dim', 2)

        freeze_layers = getattr(args, 'freeze_bert_layers', 0)

        tet_heads = getattr(args, 'tet_heads', 4)

        tet_layers = getattr(args, 'tet_layers', 2)

        tet_attn_dropout = getattr(args, 'tet_attn_dropout', 0.1)

        tet_ff_dropout = getattr(args, 'tet_ff_dropout', 0.1)

        fusion_heads = getattr(args, 'fusion_heads', 4)

        fusion_dropout = getattr(args, 'fusion_dropout', 0.1)



        self.text_model = None

        self.text_proj = None

        if self.use_text:

            self.text_model = BertTextEncoder(use_finetune=args.use_finetune,

                                              transformers=args.transformers,

                                              pretrained=args.pretrained,

                                              freeze_layers=freeze_layers)

            text_hidden = getattr(self.text_model.model.config, 'hidden_size', args.feature_dims[0])

            self.text_proj = nn.Sequential(

                nn.LayerNorm(text_hidden),

                nn.Linear(text_hidden, self.d_model),

                nn.GELU(),

                nn.Dropout(self.encoder_dropout)

            )

        audio_in = args.feature_dims[1]

        video_in = args.feature_dims[2]

        audio_hidden = getattr(args, 'audio_hidden_dim', 128)

        video_hidden = getattr(args, 'video_hidden_dim', 128)

        audio_kernel = getattr(args, 'audio_kernel_size', 3)

        video_kernel = getattr(args, 'video_kernel_size', 3)

        self.audio_encoder = TemporalProjector(audio_in, audio_hidden, self.d_model,

                                               num_layers=2, dropout=self.encoder_dropout,

                                               kernel_size=audio_kernel) if self.use_audio else None

        self.video_encoder = TemporalProjector(video_in, video_hidden, self.d_model,

                                               num_layers=2, dropout=self.encoder_dropout,

                                               kernel_size=video_kernel) if self.use_video else None



        cross_blocks = {}

        if self.use_audio:

            cross_blocks['audio'] = TextEnhancedBlock(self.d_model, tet_layers, tet_heads,

                                                      tet_attn_dropout, tet_ff_dropout)

        if self.use_video:

            cross_blocks['vision'] = TextEnhancedBlock(self.d_model, tet_layers, tet_heads,

                                                       tet_attn_dropout, tet_ff_dropout)

        if self.use_text:

            cross_blocks['text'] = TextEnhancedBlock(self.d_model, tet_layers, tet_heads,

                                                     tet_attn_dropout, tet_ff_dropout)

        if not cross_blocks:

            raise ValueError("At least one modality must be enabled.")

        self.cross_blocks = nn.ModuleDict(cross_blocks)

        modalities = []

        if self.use_audio:

            modalities.append('audio')

        if self.use_video:

            modalities.append('vision')

        if self.use_text:

            modalities.append('text')

        self.modality_order = tuple(modalities)

        self.tri_encoder = TriModalEncoder(self.d_model, fusion_heads, fusion_dropout)



        concat_dim = self.d_model * len(self.modality_order)

        self.shared_mlp = nn.Sequential(

            nn.LayerNorm(concat_dim),

            nn.Linear(concat_dim, self.fusion_dim),

            nn.GELU(),

            nn.Dropout(self.dropout)

        )



        valence_vec = getattr(args, 'emotion_valence', None)

        if valence_vec is not None:

            valence_tensor = torch.tensor(valence_vec)

        else:

            valence_tensor = None

        if self.enable_conflict:

            conf_weight_extras = {

                'l1': getattr(args, 'conf_weight_l1', 0.2),

                'conf_gap': getattr(args, 'conf_weight_conf_gap', 0.2),

                'margin_gap': getattr(args, 'conf_weight_margin_gap', 0.2)

            }

            self.conflict_module = ConflictDistribution(

                fusion_dim=self.fusion_dim,

                num_emotions=self.num_emotions,

                temperature_exp=getattr(args, 'temperature_exp', 1.0),

                temperature_imp=getattr(args, 'temperature_imp', 1.0),

                alpha_init=getattr(args, 'conf_alpha', 0.5),

                dropout=self.dropout,

                weight_js=getattr(args, 'conf_weight_js', 1.0),

                weight_pol=getattr(args, 'conf_weight_pol', 0.6),

                weight_hard=getattr(args, 'conf_weight_hard', 0.5),

                weight_entropy=getattr(args, 'conf_weight_entropy', 0.3),

                weight_margin=getattr(args, 'conf_weight_margin', 0.2),

                bias=getattr(args, 'conf_weight_bias', 0.0),

                valence=valence_tensor,

                feature_keep=getattr(args, 'conf_keep_features', None),

                feature_drop=getattr(args, 'conf_drop_features', None),

                feature_add=getattr(args, 'conf_add_features', None),

                extra_weight_inits=conf_weight_extras

            )

        else:

            self.conflict_module = None

        triad_dim = getattr(args, 'triad_dim', 128)

        triad_heads = getattr(args, 'triad_heads', 4)

        if self.enable_triad:

            self.emotion_triad = EmotionTriadFusion(

                num_emotions=self.num_emotions,

                fusion_dim=self.fusion_dim,

                affect_dim=self.affect_dim,

                triad_dim=triad_dim,

                heads=triad_heads,

                dropout=self.dropout

            )

        else:

            self.emotion_triad = None



        shared_dim = getattr(args, 'shared_hidden_dim', self.fusion_dim)

        self.head_prep = nn.Sequential(

            nn.LayerNorm(self.fusion_dim),

            nn.Linear(self.fusion_dim, shared_dim),

            nn.GELU(),

            nn.Dropout(self.dropout)

        )

        self.sarcasm_head = nn.Linear(shared_dim, 1)

        self.reg_head = nn.Linear(shared_dim, 1)

        self.explicit_head = nn.Linear(shared_dim, self.num_emotions)

        self.implicit_head = nn.Linear(shared_dim, self.num_emotions)

        self.affect_head = nn.Sequential(

            nn.LayerNorm(self.fusion_dim * 2),

            nn.Linear(self.fusion_dim * 2, self.fusion_dim),

            nn.GELU(),

            nn.Dropout(self.dropout),

            nn.Linear(self.fusion_dim, self.affect_dim)

        )



        self.audio_cls = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.vision_cls = nn.Parameter(torch.zeros(1, 1, self.d_model))

        nn.init.trunc_normal_(self.audio_cls, std=0.02)

        nn.init.trunc_normal_(self.vision_cls, std=0.02)



    def _prepare_audio(self, audio: torch.Tensor, lengths: Optional[torch.Tensor]):

        audio_tokens, mask = self.audio_encoder(audio, lengths)

        cls = self.audio_cls.expand(audio_tokens.size(0), -1, -1)

        tokens = torch.cat([cls, audio_tokens], dim=1)

        if mask is not None:

            pad = torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool)

            mask = torch.cat([pad, mask], dim=1)

        return tokens, mask

    @staticmethod

    def _valid_ratio(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if mask is None:

            return None

        if mask.dtype == torch.bool:

            valid = (~mask).float()

        else:

            valid = 1 - mask.float()

        return valid.sum(dim=-1) / valid.size(-1)



    def _modal_confidence(self, tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:

        cls_feat = tokens[:, 0, :]

        energy = torch.norm(cls_feat, dim=-1) / math.sqrt(max(cls_feat.size(-1), 1))

        ratio = self._valid_ratio(mask)

        if ratio is not None:

            energy = energy * ratio

        return energy





    def _prepare_vision(self, vision: torch.Tensor, lengths: Optional[torch.Tensor]):

        vision_tokens, mask = self.video_encoder(vision, lengths)

        cls = self.vision_cls.expand(vision_tokens.size(0), -1, -1)

        tokens = torch.cat([cls, vision_tokens], dim=1)

        if mask is not None:

            pad = torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool)

            mask = torch.cat([pad, mask], dim=1)

        return tokens, mask



    def forward(self, text, audio, vision, affect_vector=None, teacher_logits: Optional[Dict[str, torch.Tensor]] = None,

                enable_conflict: bool = True):

        text_tokens = None

        text_padding = None

        if self.use_text and text is not None:

            text_mask = text[:, 1, :].float()

            text_hidden = self.text_model(text)

            text_tokens = self.text_proj(text_hidden)

            text_padding = (text_mask == 0)

        if isinstance(audio, (tuple, list)):

            audio_x, audio_lengths = audio

        else:

            audio_x, audio_lengths = audio, None


        if isinstance(vision, (tuple, list)):

            vision_x, vision_lengths = vision

        else:

            vision_x, vision_lengths = vision, None


        tokens_map = {}

        masks_map = {}

        if self.use_audio and audio_x is not None:

            audio_tokens, audio_mask = self._prepare_audio(audio_x, audio_lengths)

            tokens_map['audio'] = audio_tokens

            masks_map['audio'] = audio_mask

        if self.use_video and vision_x is not None:

            vision_tokens, vision_mask = self._prepare_vision(vision_x, vision_lengths)

            tokens_map['vision'] = vision_tokens

            masks_map['vision'] = vision_mask

        if self.use_text and text_tokens is not None:

            tokens_map['text'] = text_tokens

            masks_map['text'] = text_padding

        modalities_active = list(self.modality_order)

        missing_modalities = [name for name in modalities_active if name not in tokens_map]

        if missing_modalities:

            raise ValueError(f"Missing inputs for modalities: {missing_modalities}")

        conf_values = [

            self._modal_confidence(tokens_map[name], masks_map.get(name))

            for name in modalities_active

        ]

        conf_stack = torch.stack(conf_values, dim=-1)

        mean_conf = conf_stack.mean(dim=0)

        key_idx = int(torch.argmax(mean_conf).item())

        key_name = modalities_active[key_idx]

        context_tokens = tokens_map[key_name]

        context_mask = masks_map.get(key_name)

        enhanced = {}

        for name in modalities_active:

            if name == key_name:

                enhanced[name] = tokens_map[name]

            else:

                enhanced[name] = self.cross_blocks[name](tokens_map[name], context_tokens, context_mask)

        ordered_blocks = [enhanced[name] for name in modalities_active]

        tri_tokens = torch.cat(ordered_blocks, dim=1)

        tri_mask = None

        if any(masks_map.get(name) is not None for name in modalities_active):

            batch = tri_tokens.size(0)

            def _ensure(mask, length):

                if mask is not None:

                    return mask

                return torch.zeros(batch, length, device=tri_tokens.device, dtype=torch.bool)

            mask_blocks = []

            for name, block in zip(modalities_active, ordered_blocks):

                mask_blocks.append(_ensure(masks_map.get(name), block.size(1)))

            tri_mask = torch.cat(mask_blocks, dim=1)

        encoded = self.tri_encoder(tri_tokens, tri_mask)

        cls_map = {}

        offset = 0

        for name, block in zip(modalities_active, ordered_blocks):

            cls_map[name] = encoded[:, offset, :]

            offset += block.size(1)

        cls_audio = cls_map.get('audio')

        cls_vision = cls_map.get('vision')

        cls_text = cls_map.get('text')

        if cls_audio is None:

            cls_audio = encoded.new_zeros(encoded.size(0), self.d_model)

        if cls_vision is None:

            cls_vision = encoded.new_zeros(encoded.size(0), self.d_model)

        if cls_text is None:

            cls_text = encoded.new_zeros(encoded.size(0), self.d_model)

        concat_cls = torch.cat([cls_map[name] for name in modalities_active], dim=-1)

        fusion_feat = self.shared_mlp(concat_cls)




        # Regression-shortcut: use dedicated reg head, skip conflict/triad/emotion losses

        if self.use_reg_head and not self.is_classification:

            head_input = self.head_prep(fusion_feat)

            explicit_logits = fusion_feat.new_zeros(fusion_feat.size(0), self.num_emotions)

            implicit_logits = fusion_feat.new_zeros(fusion_feat.size(0), self.num_emotions)

            conf_score = fusion_feat.new_zeros(fusion_feat.size(0))

            conf_stats = {}

            triad_attn = None

            affect_pred = None

            sarcasm_logits = self.reg_head(head_input).squeeze(-1)

            logits_for_metrics = sarcasm_logits.unsqueeze(-1)

        else:

            head_pre = self.head_prep(fusion_feat)

            pre_explicit = self.explicit_head(head_pre)

            pre_implicit = self.implicit_head(head_pre)

            if self.conflict_module is not None and enable_conflict:

                fusion_after_conflict, conf_score, conf_stats = self.conflict_module(

                    fusion_feat,

                    pre_explicit,

                    pre_implicit,

                    teacher=teacher_logits,

                    enable=enable_conflict

                )

            else:

                fusion_after_conflict, conf_score, conf_stats = fusion_feat, fusion_feat.new_zeros(fusion_feat.size(0)), {}



            mid_head = self.head_prep(fusion_after_conflict)

            mid_explicit = self.explicit_head(mid_head)

            mid_implicit = self.implicit_head(mid_head)



            if affect_vector is None:

                affect_vec = fusion_after_conflict.new_zeros(fusion_after_conflict.size(0), self.affect_dim)

            else:

                affect_vec = affect_vector

                if affect_vec.size(-1) > self.affect_dim:

                    affect_vec = affect_vec[:, :self.affect_dim]

                elif affect_vec.size(-1) < self.affect_dim:

                    pad = fusion_after_conflict.new_zeros(affect_vec.size(0), self.affect_dim - affect_vec.size(-1))

                    affect_vec = torch.cat([affect_vec, pad], dim=-1)



            if self.emotion_triad is not None:

                triad_feat, triad_attn, triad_summary = self.emotion_triad(

                    fusion_after_conflict, mid_explicit, mid_implicit, affect_vec

                )

            else:

                triad_feat = fusion_after_conflict

                triad_attn = None

                triad_summary = fusion_after_conflict.new_zeros(fusion_after_conflict.size(0), self.fusion_dim)

            head_input = self.head_prep(triad_feat)

            explicit_logits = self.explicit_head(head_input)

            implicit_logits = self.implicit_head(head_input)

            sarcasm_logits = self.sarcasm_head(head_input).squeeze(-1)

            if self.is_classification:

                sarcasm_prob = torch.sigmoid(sarcasm_logits)

                logits_for_metrics = torch.stack([1 - sarcasm_prob, sarcasm_prob], dim=-1)

            else:

                logits_for_metrics = sarcasm_logits.unsqueeze(-1)

            affect_pred = self.affect_head(torch.cat([triad_feat, triad_summary], dim=-1))



        return {

            'sarcasm_logits': sarcasm_logits,

            'explicit_logits': explicit_logits,

            'implicit_logits': implicit_logits,

            'explicit_probs': torch.softmax(explicit_logits, dim=-1),

            'implicit_probs': torch.softmax(implicit_logits, dim=-1),

            'conflict_score': conf_score,

            'conflict_stats': conf_stats,

            'triad_attn': triad_attn,

            'affect_pred': affect_pred,

            'Feature_f': head_input,

            'Feature_t': cls_text,

            'Feature_a': cls_audio,

            'Feature_v': cls_vision,

            'M': logits_for_metrics

        }





