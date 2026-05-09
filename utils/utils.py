"""
utils/utils.py - MODIFIED.
Keeps upstream verbatim. ADDS collate_transformer_chorus and
get_split_loader_chorus at the bottom for multi-encoder mode.
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import (DataLoader, Sampler, WeightedRandomSampler,
                               RandomSampler, SequentialSampler, sampler)
import torch.optim as optim
import math
from itertools import islice
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# UPSTREAM (verbatim)
# ============================================================

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_tranformer(batch):
    img_s = torch.cat([item[0] for item in batch], dim=0)
    img_l = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    return [img_s, img_l, label]


def get_simple_loader(dataset, batch_size=1, num_workers=1, mode='clam'):
    if mode == 'transformer':
        collate = collate_tranformer
    kwargs = {'num_workers': 0, 'pin_memory': False} \
        if device.type == "cuda" else {}
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler.SequentialSampler(dataset),
                      collate_fn=collate, **kwargs)


def get_split_loader(split_dataset, training=False, testing=False,
                     weighted=False, mode='clam'):
    if mode == 'transformer':
        collate = collate_tranformer
    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                return DataLoader(split_dataset, batch_size=1,
                                  sampler=WeightedRandomSampler(weights, len(weights)),
                                  collate_fn=collate, **kwargs)
            return DataLoader(split_dataset, batch_size=1,
                              sampler=RandomSampler(split_dataset),
                              collate_fn=collate, **kwargs)
        return DataLoader(split_dataset, batch_size=1,
                          sampler=SequentialSampler(split_dataset),
                          collate_fn=collate, **kwargs)
    ids = np.random.choice(np.arange(len(split_dataset)),
                           size=max(1, int(len(split_dataset) * 0.1)),
                           replace=False)
    return DataLoader(split_dataset, batch_size=1,
                      sampler=SubsetSequentialSampler(ids),
                      collate_fn=collate, **kwargs)


def get_optim(model, args):
    if args.opt == "adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, momentum=0.9, weight_decay=args.reg)
    raise NotImplementedError


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)
    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids, all_test_ids, sampled_train_ids = [], [], []
        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)
        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
            remaining_ids = np.setdiff1d(possible_indices, val_ids)
            all_val_ids.extend(val_ids)
            if custom_test_ids is None:
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)
            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])
        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    return 1. - Y_hat.float().eq(Y.float()).float().mean().item()


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c])
                        for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ============================================================
# NEW: CHORUS multi-encoder collate + loader
# ============================================================

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_transformer_chorus(batch):
    """
    Multi-encoder collate. Each item from Generic_MIL_Dataset_Multi is
    (feats_list, feats_list, label) where feats_list is K_v tensors
    of shape (n_patches, d_k).

    CHORUS uses batch_size=1; we unwrap and add a leading batch dim
    so each element has shape (1, n_patches, d_k).
    """
    assert len(batch) == 1, "CHORUS uses batch_size=1"
    feats_s_list, feats_l_list, label = batch[0]
    img_s_list = [f.unsqueeze(0) for f in feats_s_list]
    img_l_list = [f.unsqueeze(0) for f in feats_l_list]
    label_t = torch.LongTensor([label])
    return [img_s_list, img_l_list, label_t]


def get_split_loader_chorus(split_dataset, training=False, testing=False,
                           weighted=False, num_workers=0):
    """Same as get_split_loader but with multi-encoder collate."""
    kwargs = {'num_workers': num_workers, 'pin_memory': True, 'persistent_workers': (num_workers > 0)} if device.type == "cuda" else {}
    collate = collate_transformer_chorus
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                return DataLoader(split_dataset, batch_size=1,
                                  sampler=WeightedRandomSampler(weights, len(weights)),
                                  collate_fn=collate, **kwargs)
            return DataLoader(split_dataset, batch_size=1,
                              sampler=RandomSampler(split_dataset),
                              collate_fn=collate, **kwargs)
        return DataLoader(split_dataset, batch_size=1,
                          sampler=SequentialSampler(split_dataset),
                          collate_fn=collate, **kwargs)
    ids = np.random.choice(np.arange(len(split_dataset)),
                           size=max(1, int(len(split_dataset) * 0.1)),
                           replace=False)
    return DataLoader(split_dataset, batch_size=1,
                      sampler=SubsetSequentialSampler(ids),
                      collate_fn=collate, **kwargs)
