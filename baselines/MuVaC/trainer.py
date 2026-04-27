import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger("MMSA")


class MuVaC:
    def __init__(self, args):
        self.args = args
        self.train_mode = getattr(args, "train_mode", "classification")
        self.is_cls = str(self.train_mode).lower() == "classification"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.main_criterion = nn.CrossEntropyLoss() if self.is_cls else nn.L1Loss()

        self.lambda_kl = float(getattr(args, "lambda_kl", 0.01))
        self.lambda_rec = float(getattr(args, "lambda_rec", 0.2))
        self.lambda_ctr = float(getattr(args, "lambda_ctr", 0.2))
        self.kl_warmup_epochs = int(getattr(args, "kl_warmup_epochs", 5))
        self.temperature = float(getattr(args, "temperature", 0.07))

    @staticmethod
    def _to_class_labels(labels):
        labels = labels.view(-1)
        if torch.any(labels < 0):
            return (labels > 0).long()
        return labels.long()

    def _prepare_labels(self, labels):
        if self.is_cls:
            return self._to_class_labels(labels)
        return labels.view(-1, 1)

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        # z1,z2: [B, D], already normalized
        logits = torch.matmul(z1, z2.t()) / temperature
        target = torch.arange(z1.size(0), device=z1.device)
        loss_12 = F.cross_entropy(logits, target)
        loss_21 = F.cross_entropy(logits.t(), target)
        return 0.5 * (loss_12 + loss_21)

    def _contrastive_loss(self, outputs):
        z_t, z_a, z_v, z_lat = outputs["z_t"], outputs["z_a"], outputs["z_v"], outputs["z_lat"]
        loss_ta = self._nt_xent(z_t, z_a, self.temperature)
        loss_tv = self._nt_xent(z_t, z_v, self.temperature)
        loss_av = self._nt_xent(z_a, z_v, self.temperature)
        loss_tz = self._nt_xent(z_t, z_lat, self.temperature)
        loss_az = self._nt_xent(z_a, z_lat, self.temperature)
        loss_vz = self._nt_xent(z_v, z_lat, self.temperature)
        return (loss_ta + loss_tv + loss_av + loss_tz + loss_az + loss_vz) / 6.0

    def _compute_loss(self, outputs, labels, epoch):
        main_pred = outputs["M"]
        loss_main = self.main_criterion(main_pred, labels)

        # KL annealing for stable training
        anneal = min(1.0, float(epoch) / max(self.kl_warmup_epochs, 1))
        loss_kl = outputs.get("KL", torch.tensor(0.0, device=main_pred.device))
        loss_rec = outputs.get("REC", torch.tensor(0.0, device=main_pred.device))
        loss_ctr = self._contrastive_loss(outputs)

        total = loss_main + anneal * self.lambda_kl * loss_kl + self.lambda_rec * loss_rec + self.lambda_ctr * loss_ctr
        aux = {
            "Main": float(loss_main.detach().cpu()),
            "KL": float(loss_kl.detach().cpu()),
            "REC": float(loss_rec.detach().cpu()),
            "CTR": float(loss_ctr.detach().cpu()),
            "Anneal": anneal,
        }
        return total, aux

    def _build_optimizer(self, model):
        lr = float(getattr(self.args, "learning_rate", 1e-4))
        wd = float(getattr(self.args, "weight_decay", 1e-4))
        if not hasattr(model.Model, "text_model"):
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        bert_params = list(model.Model.text_model.parameters())
        bert_ids = {id(p) for p in bert_params}
        other_params = [p for p in model.parameters() if id(p) not in bert_ids]
        lr_bert = float(getattr(self.args, "learning_rate_bert", lr * 0.3))
        lr_other = float(getattr(self.args, "learning_rate_other", lr))

        return optim.AdamW(
            [
                {"params": bert_params, "lr": lr_bert, "weight_decay": wd},
                {"params": other_params, "lr": lr_other, "weight_decay": wd},
            ]
        )

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = self._build_optimizer(model)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min" if self.args.KeyEval == "Loss" else "max",
            factor=0.5,
            patience=int(getattr(self.args, "patience", 3)),
            verbose=True,
        )

        epochs, best_epoch = 0, 0
        min_or_max = "min" if self.args.KeyEval in ["Loss"] else "max"
        best_valid = 1e8 if min_or_max == "min" else 0.0
        epoch_results = {"train": [], "valid": [], "test": []} if return_epoch_results else None

        while True:
            epochs += 1
            model.train()
            y_pred, y_true = [], []
            total_loss = 0.0
            aux_logs = {"Main": 0.0, "KL": 0.0, "REC": 0.0, "CTR": 0.0}
            batch_cnt = 0

            with tqdm(dataloader["train"], desc=f"MuVaC Train {epochs}") as td:
                for batch_data in td:
                    vision = batch_data["vision"].to(self.args.device)
                    audio = batch_data["audio"].to(self.args.device)
                    text = batch_data["text"].to(self.args.device)
                    labels = self._prepare_labels(batch_data["labels"]["M"].to(self.args.device))

                    optimizer.zero_grad()
                    outputs = model(text, audio, vision)
                    loss, aux = self._compute_loss(outputs, labels, epochs)
                    loss.backward()

                    grad_clip = float(getattr(self.args, "grad_clip", 1.0))
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    total_loss += loss.item()
                    batch_cnt += 1
                    for k in aux_logs:
                        aux_logs[k] += aux[k]

                    y_pred.append(outputs["M"].detach().cpu())
                    y_true.append(labels.detach().cpu())
                    td.set_postfix(loss=total_loss / max(batch_cnt, 1))

            train_loss = total_loss / max(batch_cnt, 1)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            train_results["Loss"] = round(train_loss, 4)
            for k, v in aux_logs.items():
                train_results[k] = round(v / max(batch_cnt, 1), 4)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model, dataloader["valid"], mode="VAL", epoch=epochs)
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(cur_valid)

            is_better = cur_valid <= (best_valid - 1e-6) if min_or_max == "min" else cur_valid >= (best_valid + 1e-6)
            if is_better:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(val_results)
                epoch_results["test"].append(self.do_test(model, dataloader["test"], mode="TEST", epoch=epochs))

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, epoch=1):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        batch_cnt = 0
        aux_logs = {"Main": 0.0, "KL": 0.0, "REC": 0.0, "CTR": 0.0}

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader, desc=f"MuVaC {mode}") as td:
                for batch_data in td:
                    vision = batch_data["vision"].to(self.args.device)
                    audio = batch_data["audio"].to(self.args.device)
                    text = batch_data["text"].to(self.args.device)
                    labels = self._prepare_labels(batch_data["labels"]["M"].to(self.args.device))

                    outputs = model(text, audio, vision)
                    loss, aux = self._compute_loss(outputs, labels, epoch)
                    eval_loss += loss.item()
                    batch_cnt += 1
                    for k in aux_logs:
                        aux_logs[k] += aux[k]

                    y_pred.append(outputs["M"].cpu())
                    y_true.append(labels.cpu())

                    if return_sample_results:
                        ids.extend(batch_data["id"])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss / max(batch_cnt, 1), 4)
        for k, v in aux_logs.items():
            eval_results[k] = round(v / max(batch_cnt, 1), 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results["Features"] = features
            eval_results["Labels"] = all_labels

        return eval_results

