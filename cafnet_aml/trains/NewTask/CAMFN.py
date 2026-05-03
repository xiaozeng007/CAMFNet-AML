"""
Enhanced trainer for CAMFN with TET fusion, distribution-level conflict
modeling, consistency/contrastive objectives, and EMA supervision.
"""
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class CAMFN:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.train_mode = getattr(args, 'train_mode', 'classification')
        self.is_classification = str(self.train_mode).lower() == 'classification'
        self.use_reg_head = getattr(args, 'use_reg_head', not self.is_classification)
        self.use_text = bool(getattr(args, 'use_text', True))
        self.use_video = bool(getattr(args, 'use_video', True))
        self.enable_conflict = getattr(args, 'enable_conflict', self.is_classification)
        self.enable_triad = getattr(args, 'enable_triad', self.is_classification)
        self.grad_clip = getattr(args, 'grad_clip', 1.0)
        self.use_huber = getattr(args, 'use_huber_loss', False)
        self.huber_beta = getattr(args, 'huber_beta', 1.0)
        self.lambda_exp = getattr(args, 'lambda_exp', getattr(args, 'lambda_emotion', 0.2))
        self.lambda_imp = getattr(args, 'lambda_imp', getattr(args, 'lambda_emotion', 0.2))
        self.lambda_cons = getattr(args, 'lambda_cons', getattr(args, 'lambda_conflict', 0.5))
        self.lambda_cons_main = getattr(args, 'lambda_cons_main', 0.3)
        self.lambda_ctr = getattr(args, 'lambda_ctr', 0.5)
        self.lambda_affect = getattr(args, 'lambda_affect', 0.1)
        self.cold_epochs = getattr(args, 'cold_start_epochs', 2)
        self.consistency_start = getattr(args, 'consistency_start_epoch', 2)
        self.contrast_start = getattr(args, 'contrast_start_epoch', 4)
        self.contrast_margin = getattr(args, 'contrast_margin', 0.5)
        self.contrast_temp = getattr(args, 'contrast_temperature', 1.0)
        self.use_ema_teacher = getattr(args, 'use_ema_teacher', True)
        self.ema_decay = getattr(args, 'ema_decay', 0.999)
        self.use_focal = getattr(args, 'use_focal_loss', False)
        self.focal_gamma = getattr(args, 'focal_gamma', 2.0)
        self.unfreeze_start = getattr(args, 'unfreeze_start_epoch', 3)
        self.unfreeze_step = getattr(args, 'unfreeze_step_layers', 2)
        self.frozen_layers = getattr(args, 'freeze_bert_layers', 0)
        self.sarcasm_criterion = None
        if self.use_huber:
            self.regression_criterion = nn.HuberLoss(delta=self.huber_beta, reduction='mean')
        else:
            self.regression_criterion = nn.SmoothL1Loss(reduction='mean')
        self.pos_weight_tensor = None
        self.ema_model = None

    def _augment_regression_metrics(self, metrics, preds, truths):
        """Ensure regression runs always log MAE/Corr alongside Loss."""
        if self.is_classification:
            return metrics
        preds_np = preds.view(-1).cpu().numpy()
        truths_np = truths.view(-1).cpu().numpy()
        if preds_np.shape[0] == 0:
            return metrics
        mae = float(np.mean(np.abs(preds_np - truths_np)))
        corr = 0.0
        if preds_np.shape[0] > 1:
            try:
                corr = float(np.corrcoef(preds_np, truths_np)[0][1])
            except Exception:
                corr = 0.0
        metrics.setdefault('MAE', round(mae, 4))
        metrics.setdefault('Corr', round(corr, 4))
        return metrics

    def _estimate_pos_weight(self, dataset):
        labels = dataset.labels['M']
        labels = np.array(labels)
        pos = np.sum(labels > 0)
        neg = np.sum(labels <= 0)
        if pos == 0:
            return 1.0
        return max(neg / pos, 1.0)

    def _maybe_init_sarcasm_criterion(self, dataloader):
        if not self.is_classification:
            return
        if self.sarcasm_criterion is not None:
            return
        pos_weight = getattr(self.args, 'pos_weight', None)
        if pos_weight is None:
            pos_weight = self._estimate_pos_weight(dataloader['train'].dataset)
        weight_tensor = torch.tensor(pos_weight, device=self.device)
        self.pos_weight_tensor = weight_tensor
        self.sarcasm_criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def _build_optimizer(self, model):
        base_model = model.Model
        text_encoder = getattr(base_model, 'text_model', None)
        if self.use_text and text_encoder is not None:
            bert_params = [p for p in text_encoder.parameters() if p.requires_grad]
        else:
            bert_params = []
        bert_param_ids = {id(p) for p in bert_params}
        other_params = [p for p in base_model.parameters() if id(p) not in bert_param_ids]
        param_groups = []
        if bert_params:
            param_groups.append({
                'params': bert_params,
                'lr': self.args.learning_rate_bert,
                'weight_decay': getattr(self.args, 'weight_decay_bert', 0.01)
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': getattr(self.args, 'learning_rate_other', 1e-4),
                'weight_decay': getattr(self.args, 'weight_decay_other', 0.01)
            })
        optimizer = AdamW(param_groups)
        return optimizer

    def _build_scheduler(self, optimizer, steps_per_epoch, epochs):
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_ratio = getattr(self.args, 'warmup_ratio', 0.1)
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    def _init_ema(self, model):
        if not self.use_ema_teacher:
            self.ema_model = None
            return
        self.ema_model = copy.deepcopy(model)
        self.ema_model.to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update_ema(self, model):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _get_teacher_logits(self, text, audio, vision, affect, enable_conflict):
        if self.ema_model is None or not enable_conflict:
            return None
        self.ema_model.eval()
        with torch.no_grad():
            outputs = self.ema_model(text, audio, vision, affect_vector=affect, enable_conflict=False)
        return {
            'explicit': outputs['explicit_logits'].detach(),
            'implicit': outputs['implicit_logits'].detach()
        }

    def _apply_progressive_unfreeze(self, model, epoch):
        if epoch < self.unfreeze_start or self.frozen_layers <= 0:
            return
        text_encoder = getattr(model.Model, 'text_model', None)
        if (not self.use_text) or text_encoder is None or not hasattr(text_encoder, 'set_freeze_layers'):
            return
        epochs_since = epoch - self.unfreeze_start + 1
        new_freeze = max(0, self.frozen_layers - epochs_since * self.unfreeze_step)
        text_encoder.set_freeze_layers(new_freeze)

    def _prepare_lengths(self, tensor):
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            return tensor.to(self.device).long()
        return torch.tensor(tensor, device=self.device).long()

    def _prepare_batch(self, batch):
        text = batch.get('text', None)
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)
        else:
            text = None
        audio = batch.get('audio', None)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(self.device)
        else:
            audio = None
        vision = batch.get('vision', None)
        if isinstance(vision, torch.Tensor):
            vision = vision.to(self.device)
        else:
            vision = None
        audio_lengths = self._prepare_lengths(batch.get('audio_lengths', None))
        vision_lengths = self._prepare_lengths(batch.get('vision_lengths', None))
        audio_input = (audio, audio_lengths) if (audio is not None and audio_lengths is not None) else audio
        if vision is not None and vision_lengths is not None:
            vision_input = (vision, vision_lengths)
        else:
            vision_input = vision

        labels = batch['labels']['M'].view(-1).to(self.device).float()
        contrast_labels = (labels > 0).float()
        main_labels = contrast_labels if self.is_classification else labels
        explicit_labels = batch.get('explicit_label', None)
        if explicit_labels is not None:
            explicit_labels = explicit_labels.to(self.device).long()
        implicit_labels = batch.get('implicit_label', None)
        if implicit_labels is not None:
            implicit_labels = implicit_labels.to(self.device).long()
        affect = batch.get('affect_vector', None)
        if affect is not None:
            affect = affect.to(self.device).float()
        else:
            val = batch.get('valence', None)
            aro = batch.get('arousal', None)
            if val is not None and aro is not None:
                affect = torch.stack([val.to(self.device), aro.to(self.device)], dim=-1).float()
        return (text, audio_input, vision_input, affect,
                main_labels, explicit_labels, implicit_labels, contrast_labels)

    def _main_loss(self, logits, labels):
        if not self.is_classification:
            return self.regression_criterion(logits.view(-1), labels.view(-1))
        if not self.use_focal:
            return self.sarcasm_criterion(logits, labels)
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy(prob, labels, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.focal_gamma) * ce
        if self.pos_weight_tensor is not None:
            loss = torch.where(labels > 0, self.pos_weight_tensor * loss, loss)
        return loss.mean()

    def _eval_loss(self, logits, labels):
        if not self.is_classification:
            return self.regression_criterion(logits.view(-1), labels.view(-1))
        if not self.use_focal:
            return self.sarcasm_criterion(logits, labels)
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy(prob, labels, reduction='none')
        if self.pos_weight_tensor is not None:
            ce = torch.where(labels > 0, self.pos_weight_tensor * ce, ce)
        return ce.mean()

    @staticmethod
    def _symmetric_kl(logits_a, logits_b):
        p_a = torch.softmax(logits_a, dim=-1)
        p_b = torch.softmax(logits_b, dim=-1)
        log_a = torch.log_softmax(logits_a, dim=-1)
        log_b = torch.log_softmax(logits_b, dim=-1)
        kl_ab = torch.sum(p_a * (log_a - log_b), dim=-1)
        kl_ba = torch.sum(p_b * (log_b - log_a), dim=-1)
        return kl_ab + kl_ba

    @staticmethod
    def _binary_sym_kl(logits_a, logits_b):
        prob_a = torch.sigmoid(logits_a)
        prob_b = torch.sigmoid(logits_b)
        prob_a = prob_a.clamp(1e-6, 1 - 1e-6)
        prob_b = prob_b.clamp(1e-6, 1 - 1e-6)
        kl_ab = prob_a * (prob_a / prob_b).log() + (1 - prob_a) * ((1 - prob_a) / (1 - prob_b)).log()
        kl_ba = prob_b * (prob_b / prob_a).log() + (1 - prob_b) * ((1 - prob_b) / (1 - prob_a)).log()
        return kl_ab + kl_ba

    def _consistency_loss(self, out_a, out_b, conf_scores):
        sym_exp = self._symmetric_kl(out_a['explicit_logits'], out_b['explicit_logits'])
        sym_imp = self._symmetric_kl(out_a['implicit_logits'], out_b['implicit_logits'])
        weight = (1 - conf_scores.detach()).clamp(min=0.0)
        emo_loss = (weight * (sym_exp + sym_imp)).mean()
        if self.is_classification:
            main_loss = (weight * self._binary_sym_kl(out_a['sarcasm_logits'], out_b['sarcasm_logits'])).mean()
        else:
            diff = (out_a['sarcasm_logits'] - out_b['sarcasm_logits']).pow(2)
            main_loss = (weight * diff).mean()
        return emo_loss, main_loss

    def _contrastive_loss(self, outputs, labels, conf_scores):
        # labels: contrast_labels (binary) for weighting; preserves classification-style mask even in regression
        z_e = F.normalize(outputs['explicit_logits'] / self.contrast_temp, dim=-1)
        z_i = F.normalize(outputs['implicit_logits'] / self.contrast_temp, dim=-1)
        dist = torch.norm(z_e - z_i, dim=-1)
        hinge = torch.relu(self.contrast_margin - dist)
        weight = (conf_scores.detach() * labels).clamp(min=0.0)
        if torch.sum(weight) <= 1e-6:
            return hinge.mean() * 0
        return torch.sum(weight * hinge) / torch.clamp(torch.sum(weight), min=1.0)

    def _get_phase(self, epoch):
        if epoch <= self.cold_epochs:
            return 'cold'
        if epoch < self.contrast_start:
            return 'consistency'
        return 'full'

    def _train_one_epoch(self, model, dataloader, optimizer, scheduler, epoch):
        model.train()
        total_loss = 0.0
        total_batches = 0
        preds, trues = [], []
        conf_logs = []
        phase = self._get_phase(epoch)
        enable_conflict = (self.enable_conflict and phase != 'cold' and not self.use_reg_head)
        enable_emotion = (self.enable_triad and phase != 'cold' and not self.use_reg_head)
        enable_consistency = (phase != 'cold' and not self.use_reg_head
                              and (self.lambda_cons > 0 or self.lambda_cons_main > 0))
        enable_contrast = (phase == 'full' and not self.use_reg_head and self.lambda_ctr > 0)
        self._apply_progressive_unfreeze(model, epoch)

        with tqdm(dataloader, desc=f"Epoch {epoch} | Train") as td:
            for batch in td:
                optimizer.zero_grad()
                (text, audio, vision, affect,
                 main_labels, explicit_labels, implicit_labels, contrast_labels) = self._prepare_batch(batch)
                teacher_logits = self._get_teacher_logits(text, audio, vision, affect, enable_conflict)
                outputs = model(text, audio, vision, affect_vector=affect,
                                teacher_logits=teacher_logits, enable_conflict=enable_conflict)
                loss = self._main_loss(outputs['sarcasm_logits'], main_labels)

                if enable_emotion and explicit_labels is not None and implicit_labels is not None:
                    loss_exp = self.lambda_exp * F.cross_entropy(outputs['explicit_logits'], explicit_labels)
                    loss_imp = self.lambda_imp * F.cross_entropy(outputs['implicit_logits'], implicit_labels)
                    loss = loss + loss_exp + loss_imp

                if enable_consistency:
                    outputs_aug = model(text, audio, vision, affect_vector=affect,
                                        teacher_logits=teacher_logits, enable_conflict=enable_conflict)
                    loss_cons_emo, loss_cons_main = self._consistency_loss(outputs, outputs_aug, outputs['conflict_score'])
                    if self.lambda_cons > 0:
                        loss = loss + self.lambda_cons * loss_cons_emo
                    if self.lambda_cons_main > 0:
                        loss = loss + self.lambda_cons_main * loss_cons_main

                if enable_contrast and contrast_labels is not None:
                    loss_ctr = self._contrastive_loss(outputs, contrast_labels, outputs['conflict_score'])
                    loss = loss + self.lambda_ctr * loss_ctr

                if (self.lambda_affect > 0 and affect is not None
                        and outputs.get('affect_pred', None) is not None
                        and enable_emotion):
                    target_affect = affect
                    affect_loss = F.smooth_l1_loss(outputs['affect_pred'], target_affect, reduction='none')
                    weight = 1.0 + outputs['conflict_score'].detach().unsqueeze(-1)
                    affect_loss = (affect_loss * weight).mean()
                    loss = loss + self.lambda_affect * affect_loss

                loss.backward()
                clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                self._update_ema(model)

                total_loss += loss.item()
                total_batches += 1
                preds.append(outputs['M'].detach().cpu())
                trues.append(main_labels.detach().cpu().long() if self.is_classification else main_labels.detach().cpu())
                conf_logs.append(outputs['conflict_score'].detach().cpu())
                td.set_postfix(loss=total_loss / max(total_batches, 1))

        pred = torch.cat(preds)
        true = torch.cat(trues)
        metrics = self.metrics(pred, true)
        metrics = self._augment_regression_metrics(metrics, pred, true)
        metrics['Loss'] = round(total_loss / max(total_batches, 1), 4)
        if conf_logs:
            metrics['Conflict'] = round(torch.cat(conf_logs).mean().item(), 4)
        logger.info(f"TRAIN-{epoch}: >> {dict_to_str(metrics)}")
        return metrics

    def _get_eval_model(self, model):
        return self.ema_model if self.ema_model is not None else model

    def do_train(self, model, dataloader, return_epoch_results=False):
        self._maybe_init_sarcasm_criterion(dataloader)
        optimizer = self._build_optimizer(model)
        epochs = getattr(self.args, 'epochs', 30)
        scheduler = self._build_scheduler(optimizer, len(dataloader['train']), epochs)
        self._init_ema(model)
        conflict_module = getattr(model.Model, 'conflict_module', None)
        if conflict_module is not None and hasattr(conflict_module, 'active_feature_names'):
            logger.info(f"Conflict features: {list(conflict_module.active_feature_names)}")
        best_metric = -1e9
        best_epoch = 0
        early_stop_patience = getattr(self.args, 'early_stop', 5)
        epoch_results = {'train': [], 'valid': [], 'test': []} if return_epoch_results else None

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(model, dataloader['train'], optimizer, scheduler, epoch)
            eval_model = self._get_eval_model(model)
            val_results = self.do_test(eval_model, dataloader['valid'], mode="VAL")
            cur_metric = val_results[self.args.KeyEval]
            if cur_metric >= best_metric + 1e-6:
                best_metric = cur_metric
                best_epoch = epoch
                state_dict = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
                torch.save(state_dict, self.args.model_save_path)
                model.to(self.device)
                if self.ema_model is not None:
                    self.ema_model.to(self.device)
            if return_epoch_results:
                epoch_results['train'].append(dict(train_metrics))
                epoch_results['valid'].append(val_results)
                eval_snapshot = self.do_test(eval_model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(eval_snapshot)
            if epoch - best_epoch >= early_stop_patience:
                break
        return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model_to_use = self._get_eval_model(model)
        model_to_use.eval()
        preds, trues = [], []
        eval_loss = 0.0
        samples = []
        features = {
            'Feature_t': [],
            'Feature_a': [],
            'Feature_v': [],
            'Feature_f': []
        }
        with torch.no_grad():
            with tqdm(dataloader, desc=f"{mode} | Eval") as td:
                for batch in td:
                    (text, audio, vision, affect,
                     main_labels, _, _, contrast_labels) = self._prepare_batch(batch)
                    outputs = model_to_use(text, audio, vision, affect_vector=affect,
                                           enable_conflict=self.enable_conflict and not self.use_reg_head)
                    loss = self._eval_loss(outputs['sarcasm_logits'], main_labels)
                    eval_loss += loss.item()
                    preds.append(outputs['M'].cpu())
                    trues.append(main_labels.cpu().long() if self.is_classification else main_labels.cpu())
                    if return_sample_results:
                        samples.extend(batch['id'])
                        for key in features:
                            features[key].append(outputs[key].cpu().numpy())
        pred = torch.cat(preds)
        true = torch.cat(trues)
        eval_results = self.metrics(pred, true)
        eval_results = self._augment_regression_metrics(eval_results, pred, true)
        eval_results['Loss'] = round(eval_loss / max(len(dataloader), 1), 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        if return_sample_results:
            eval_results['Ids'] = samples
            for key in features:
                if features[key]:
                    features[key] = np.concatenate(features[key], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = true.numpy().tolist()
            eval_results['Preds'] = pred.numpy().tolist()
        return eval_results
