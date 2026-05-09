"""
datasets/dataset_generic.py
---------------------------
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])
    df.to_csv(filename)
    print()



class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path='dataset_csv/ccrcc_clean.csv', mode='clam',
                 shuffle=False, seed=7, print_info=True, label_dict={},
                 filter_dict={}, ignore=[], patient_strat=False,
                 label_col=None, patient_voting='max'):
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir_s = None
        self.data_dir_l = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.patient_data_prep(patient_voting)
        self.mode = mode
        self.cls_ids_prep()
        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir_s=self.data_dir_s,
                                  data_dir_l=self.data_dir_l, mode=self.mode,
                                  num_classes=self.num_classes)
        else:
            split = None
        print(len(split) if split is not None else 0)
        return split

    def return_splits(self, from_id=True, csv_path=None):
        assert csv_path
        all_splits = pd.read_csv(csv_path,
                                 dtype={'dir': str, 'case_id': str,
                                        'slide_id': str, 'label': str})
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        test_split = self.get_split_from_df(all_splits, 'test')
        return train_split, val_split, test_split

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None





    def __len__(self):
        return len(self.slide_data)


# ============================================================
# NEW: multi-encoder .h5 versions for CHORUS
# ============================================================

class Generic_MIL_Dataset_Multi(Generic_WSI_Classification_Dataset):
    """
    Multi-encoder MIL dataset reading TRIDENT-style .h5 features.
    Returns a list of K_v feature tensors per slide.

    encoder_dirs : dict {encoder_name: dir containing h5_files/<slide>.h5}
    encoder_order : list of encoder names (must match CCA fit order!)
    """

    def __init__(self, encoder_dirs, encoder_order, mode='transformer', **kwargs):
        super().__init__(**kwargs)
        assert isinstance(encoder_dirs, dict) and len(encoder_dirs) > 0
        self.encoder_dirs = encoder_dirs
        self.encoder_order = encoder_order or list(encoder_dirs.keys())
        for e in self.encoder_order:
            assert e in encoder_dirs, f"encoder {e} not in encoder_dirs"
        self.mode = mode
        self.use_h5 = True
        self.data_dir_s = self.encoder_dirs[self.encoder_order[0]]
        self.data_dir_l = self.encoder_dirs[self.encoder_order[0]]

    _GLOBAL_CACHE = {}
    
    def _load_one(self, enc, slide_id):
        cache_key = f"{enc}_{slide_id}"
        if cache_key in self._GLOBAL_CACHE:
            return self._GLOBAL_CACHE[cache_key].detach()

        # 1. Try .pt first (in pt_files/ or directly)
        pt_path = os.path.join(self.encoder_dirs[enc], 'pt_files', f'{slide_id}.pt')
        if not os.path.exists(pt_path):
            pt_path = os.path.join(self.encoder_dirs[enc], f'{slide_id}.pt')
        
        if os.path.exists(pt_path):
            f = torch.load(pt_path, weights_only=True).float()
        else:
            # 2. Fallback to .h5
            path = os.path.join(self.encoder_dirs[enc], 'h5_files', f'{slide_id}.h5')
            if not os.path.exists(path):
                path = os.path.join(self.encoder_dirs[enc], f'{slide_id}.h5')
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Neither .pt nor .h5 found for {slide_id} in {self.encoder_dirs[enc]}")

            with h5py.File(path, 'r') as h5f:
                if 'features' in h5f:
                    f = h5f['features'][:]
                elif 'feats' in h5f:
                    f = h5f['feats'][:]
                else:
                    raise KeyError(f"No 'features'/'feats' key in {path}")
            f = torch.as_tensor(f).float()
        
        # Cache on CPU (multiprocessing safe)
        self._GLOBAL_CACHE[cache_key] = f.detach()
        return f.detach()

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]

        feats_per_encoder = []
        n_ref = None
        for enc in self.encoder_order:
            f = self._load_one(enc, slide_id)
            feats_per_encoder.append(f)
            if n_ref is None:
                n_ref = f.shape[0]
            elif f.shape[0] != n_ref:
                raise RuntimeError(
                    f"Patch count mismatch for slide {slide_id}: "
                    f"{self.encoder_order[0]}={n_ref}, {enc}={f.shape[0]}. "
                    "Re-tile to identical coordinates across encoders."
                )

        # Return (x_s_list, x_l_list, label)
        return feats_per_encoder, feats_per_encoder, label

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split_Multi(
                df_slice,
                encoder_dirs=self.encoder_dirs,
                encoder_order=self.encoder_order,
                mode=self.mode,
                num_classes=self.num_classes,
            )
        else:
            split = None
        print(len(split) if split is not None else 0)
        return split


class Generic_Split_Multi(Generic_MIL_Dataset_Multi):
    def __init__(self, slide_data, encoder_dirs, encoder_order,
                 mode='transformer', num_classes=2):
        self.use_h5 = True
        self.slide_data = slide_data
        self.encoder_dirs = encoder_dirs
        self.encoder_order = encoder_order
        self.data_dir_s = encoder_dirs[encoder_order[0]]
        self.data_dir_l = encoder_dirs[encoder_order[0]]
        self.mode = mode
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
