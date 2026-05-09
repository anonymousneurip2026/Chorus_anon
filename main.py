"""
main.py - MODIFIED.

Changes vs upstream:
- All hardcoded csv_path '/home/gzr/EasyMIL/dataset_csv/...' replaced with
  local 'dataset_csv/...'
- 'CHORUS' added to --model_type choices
- New CHORUS CLI flags (cca_visual_path, cca_text_path, chorus_variant,
  chorus_alpha_init, chorus_beta_init, chorus_tau, encoder_dir, encoder_order,
  disable_*)
- Branches dataset construction on model_type: CHORUS uses
  Generic_MIL_Dataset_Multi (multi-encoder .h5)
"""

from __future__ import print_function
import argparse
import os
import torch
import pandas as pd
import numpy as np

from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset_Multi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None)
parser.add_argument('--data_folder_s', type=str, default=None)
parser.add_argument('--data_folder_l', type=str, default=None)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--results_dir', default='./results')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False)
parser.add_argument('--model_type', type=str,
                    choices=['CHORUS'],   # CHORUS-only
                    default='CHORUS')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--exp_code', type=str)
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--sim_threshold", type=float, default=0.8)
parser.add_argument('--val_every', type=int, default=1, help='validation frequency (default: 1)')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader (default: 8)')
parser.add_argument('--patience', type=int, default=20, help='patience for early stopping (default: 20)')
parser.add_argument('--stop_epoch', type=int, default=60, help='minimum epoch to consider early stopping (default: 60)')

# ===========================================================
# CHORUS flags (NEW)
# ===========================================================
parser.add_argument('--cca_visual_path', type=str, default=None,
                    help='Path to visual_cca.npz from cca/fit_visual_cca.py')
parser.add_argument('--cca_text_path', type=str, default=None,
                    help='Path to text_cca.npz from cca/fit_text_cca.py (optional)')
parser.add_argument('--chorus_variant', type=str, default='none',
                    choices=['none', 'v1_embed', 'v2_attn', 'v3_local',
                             'v4_hybrid', 'v5_precision', 'v6_threshold',
                             'v9_inverse', 'v10_temp'])
parser.add_argument('--chorus_alpha_init', type=float, default=1.0)
parser.add_argument('--chorus_beta_init', type=float, default=1.0)
parser.add_argument('--chorus_tau', type=float, default=0.3)
parser.add_argument('--encoder_dir', action='append', default=[],
                    help='Repeatable: <name>=<dir> where <dir> contains h5_files/')
parser.add_argument('--encoder_order', nargs='+', default=None,
                    help='Explicit encoder order; MUST match CCA fit order')
# Per-stage disable flags
parser.add_argument('--disable_prestage', action='store_true', default=False)
parser.add_argument('--disable_stage1_rho', action='store_true', default=False)
parser.add_argument('--disable_stage2_sigma', action='store_true', default=False)
parser.add_argument('--disable_stage3_rho', action='store_true', default=False)
parser.add_argument('--disable_stage4_qk', action='store_true', default=False)
parser.add_argument('--disable_stage4_v', action='store_true', default=False)
parser.add_argument('--disable_stage4_sigma', action='store_true', default=False)

args = parser.parse_args()


def _parse_encoder_dirs(pairs):
    out = {}
    for p in pairs:
        if '=' not in p:
            raise ValueError(f"Bad --encoder_dir: {p} (expected name=path)")
        name, path = p.split('=', 1)
        out[name.strip()] = path.strip()
    return out


args.encoder_dirs = _parse_encoder_dirs(args.encoder_dir)
args.text_prompt = (np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
                    if args.text_prompt_path else None)


# Using seed_torch from utils.utils


seed_torch(args.seed)

settings = {
    'num_splits': args.k, 'k_start': args.k_start, 'k_end': args.k_end,
    'task': args.task, 'max_epochs': args.max_epochs,
    'results_dir': args.results_dir, 'lr': args.lr,
    'experiment': args.exp_code, 'label_frac': args.label_frac,
    'seed': args.seed, 'model_type': args.model_type, 'mode': args.mode,
    "use_drop_out": args.drop_out, 'weighted_sample': args.weighted_sample,
    'opt': args.opt, 'chorus_variant': args.chorus_variant,
}

print('\nLoad Dataset')

# Task -> label_dict + LOCAL csv_path
if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes = 3
    csv_path = 'dataset_csv/RCC.csv'
    label_dict = {'KICH': 0, 'KIRC': 1, 'KIRP': 2}
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes = 2
    csv_path = 'dataset_csv/LUAD_LUSC.csv'
    label_dict = {'LUAD': 0, 'LUSC': 1}
else:
    raise NotImplementedError(args.task)

# === Branch on model_type ===
dataset = Generic_MIL_Dataset_Multi(
    csv_path=csv_path,
    encoder_dirs=args.encoder_dirs,
    encoder_order=args.encoder_order,
    mode=args.mode,
    shuffle=False,
    print_info=True,
    label_dict=label_dict,
    patient_strat=False,
    ignore=[],
)

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
args.results_dir = os.path.join(args.results_dir,
                                 str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)
print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir), f"Splits directory does not exist: {args.split_dir}"
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}: {val}")


def main(args):
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end

    all_test_auc, all_val_auc = [], []
    all_test_acc, all_val_acc, all_test_f1 = [], [], []
    folds = np.arange(start, end)

    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path=f'{args.split_dir}/splits_{i}.csv'
        )
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(
            (train_dataset, val_dataset, test_dataset), i, args
        )
        all_test_auc.append(test_auc); all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc); all_val_acc.append(val_acc)
        save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
                              'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({
        'metric': ['mean', 'std'],
        'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
        'test_f1':  [np.mean(all_test_f1),  np.std(all_test_f1)],
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
    })
    if len(folds) != args.k:
        save_name = f'summary_partial_{folds[0]}_{folds[-1]}.csv'
        result_name = f'result_partial_{folds[0]}_{folds[-1]}.csv'
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'
    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    main(args)
    print("finished!")
    print("end script")
