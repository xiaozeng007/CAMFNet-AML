import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from run import MMSA_run


BASE5 = ['js', 'polar_gap', 'hard', 'entropy_inv', 'margin_inv']
DATASET_NAME = 'mustardpp'


def _parse_seeds(raw):
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_metric(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # "(84.63, 1.70)" -> 84.63
    if text.startswith('(') and ',' in text:
        text = text[1:text.find(',')].strip()
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _build_experiments():
    experiments = [{
        'name': 'keep5',
        'group': 'baseline',
        'cfg': {}
    }]
    for feat in BASE5:
        experiments.append({
            'name': f'drop_{feat}',
            'group': 'drop_one',
            'cfg': {'conf_drop_features': [feat]}
        })
    return experiments


def _read_one_exp_metrics(out_root: Path, dataset: str):
    csv_path = out_root / 'results' / 'normal' / f'{dataset}.csv'
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if 'Model' not in df.columns:
        return {}
    chosen = df[df['Model'] == 'camfn_ensemble']
    if chosen.empty:
        chosen = df[df['Model'] == 'camfn']
    if chosen.empty:
        return {}
    row = chosen.iloc[-1].to_dict()
    metrics = {}
    for k, v in row.items():
        if k == 'Model':
            continue
        metrics[k] = _parse_metric(v)
    return metrics


def _collect_all_records(root: Path, dataset: str):
    records = []
    if not root.exists():
        return records
    for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cfg_path = exp_dir / 'exp_config.json'
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            except Exception:
                cfg = {}
        metrics = _read_one_exp_metrics(exp_dir, dataset)
        rec = {
            'exp_name': exp_dir.name,
            'group': cfg.get('group', ''),
            'cfg': json.dumps(cfg.get('cfg', {}), ensure_ascii=False)
        }
        rec.update(metrics)
        records.append(rec)
    return records


def _write_summary(root: Path, dataset: str, key_metric: str):
    records = _collect_all_records(root, dataset)
    if not records:
        return None, None
    df = pd.DataFrame(records)
    if 'exp_name' not in df.columns:
        return None, None
    baseline_row = df[df['exp_name'] == 'keep5']
    baseline = None
    if not baseline_row.empty and key_metric in baseline_row.columns:
        baseline = baseline_row.iloc[0].get(key_metric, None)
    if key_metric in df.columns:
        df['rank_key'] = pd.to_numeric(df[key_metric], errors='coerce')
        df = df.sort_values(by='rank_key', ascending=False, na_position='last').drop(columns=['rank_key'])
    if baseline is not None and key_metric in df.columns:
        df[f'delta_{key_metric}_vs_keep5'] = pd.to_numeric(df[key_metric], errors='coerce') - float(baseline)

    summary_csv = root / 'summary_conflict_ablation.csv'
    df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    summary_md = root / 'summary_conflict_ablation.md'
    lines = []
    lines.append(f'# CAMFNet-AML Conflict Ablation Summary ({dataset})')
    lines.append('')
    lines.append(f'- key metric: `{key_metric}`')
    if baseline is not None:
        lines.append(f'- baseline keep5 `{key_metric}`: `{baseline}`')
    lines.append(f'- experiments: `{len(df)}`')
    lines.append('')
    view_cols = [c for c in ['exp_name', 'group', key_metric, f'delta_{key_metric}_vs_keep5', 'F1_score', 'Loss', 'cfg'] if c in df.columns]
    if view_cols:
        try:
            lines.append(df[view_cols].to_markdown(index=False))
        except Exception:
            lines.append('```')
            lines.append(df[view_cols].to_string(index=False))
            lines.append('```')
    summary_md.write_text('\n'.join(lines), encoding='utf-8')
    return summary_csv, summary_md


def main():
    parser = argparse.ArgumentParser(
        description='Run CAMFNet-AML 5-dim conflict ablation on MUStARD++ (keep5 + drop-one).'
    )
    parser.add_argument('--seeds', type=str, default='1111,1112,1113,1114,1115',
                        help='Comma-separated seeds.')
    parser.add_argument('--gpu-id', type=int, default=0, help='Single GPU id.')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers.')
    parser.add_argument('--root', type=str, default='_tmp_run/conflict_ablation',
                        help='Output root folder.')
    parser.add_argument('--key-metric', type=str, default='Accuracy',
                        help='Metric used for ranking in summary.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip an experiment if its result csv already exists.')
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    dataset = DATASET_NAME
    root = Path(args.root) / dataset
    root.mkdir(parents=True, exist_ok=True)
    experiments = _build_experiments()
    if torch.cuda.is_available():
        print(f'CUDA available: {torch.cuda.device_count()} GPU(s). Using gpu-id={args.gpu_id}.')
    else:
        print('Warning: CUDA is NOT available in current env, training will run on CPU.')

    print(f'Planned experiments: {len(experiments)}')
    for idx, exp in enumerate(experiments, start=1):
        print(f'[{idx:02d}] {exp["name"]} | group={exp["group"]} | cfg={exp["cfg"]}')

    for exp in experiments:
        exp_name = exp['name']
        exp_cfg = exp['cfg']
        out_root = root / exp_name
        model_dir = out_root / 'saved_models'
        res_dir = out_root / 'results'
        log_dir = out_root / 'logs'
        csv_path = res_dir / 'normal' / f'{dataset}.csv'
        if args.skip_existing and csv_path.exists():
            print(f'== Skip {exp_name} (existing: {csv_path}) ==')
            continue

        run_cfg = {
            'enable_conflict': True,
            'enable_triad': True,
            'use_reg_head': False
        }
        run_cfg.update(exp_cfg)

        out_root.mkdir(parents=True, exist_ok=True)
        cfg_path = out_root / 'exp_config.json'
        if not cfg_path.exists():
            try:
                cfg_path.write_text(json.dumps({
                    'name': exp_name,
                    'group': exp['group'],
                    'cfg': exp_cfg
                }, ensure_ascii=False, indent=2), encoding='utf-8')
            except PermissionError:
                print(f'Warn: no permission to write {cfg_path}, continue.')

        print(f'== Running {exp_name} on {dataset} ==')
        MMSA_run(
            model_name='camfnet_aml',
            dataset_name=dataset,
            config=run_cfg,
            seeds=seeds,
            gpu_ids=[args.gpu_id],
            num_workers=args.num_workers,
            verbose_level=1,
            model_save_dir=model_dir,
            res_save_dir=res_dir,
            log_dir=log_dir
        )

        metrics = _read_one_exp_metrics(out_root, dataset)
        print(f'== Done {exp_name} | metrics={metrics} ==')

    summary_csv, summary_md = _write_summary(root, dataset, args.key_metric)
    print('== Summary ==')
    print(f'csv: {summary_csv}')
    print(f'md : {summary_md}')


if __name__ == '__main__':
    main()
