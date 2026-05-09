"""
utils/core_utils.py - MODIFIED.
Adds CHORUS branch in train(), and routes batches correctly
when args.model_type == 'CHORUS'.
"""

import numpy as np
import torch
import torch.nn as nn
from utils.utils import (print_network, get_optim, get_split_loader,
                          get_split_loader_chorus, calculate_error)
import os
from datasets.dataset_generic import save_splits
# from models.model_mil import MIL_fc, MIL_fc_mc (DELETED - UNUSED)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
# from utils.loss_utils import FocalLoss (DELETED - UNUSED)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


class Accuracy_Logger(object):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat); Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int); Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        acc = None if count == 0 else float(correct) / count
        return acc, correct, count


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=60, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def _is_chorus(args):
    return getattr(args, 'model_type', None) == 'CHORUS'


def _move_to_device(batch, device, chorus):
    """Handle both standard (3-tensor) and multi-encoder (list+list+tensor) batch formats."""
    if chorus:
        img_s_list, img_l_list, label = batch
        img_s_list = [x.to(device) for x in img_s_list]
        img_l_list = [x.to(device) for x in img_l_list]
        label = label.to(device)
        return img_s_list, img_l_list, label
    else:
        data_s, data_l, label = batch
        return data_s.to(device), data_l.to(device), label.to(device)


def train(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'],
                os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")
    print(f"Testing on {len(test_split)} samples")

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'CHORUS':
        # ===== NEW BRANCH =====
        import ml_collections
        from models.model_CHORUS import CHORUS
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.max_context_length = 8192
        config.window_size = args.window_size
        config.sim_threshold = args.sim_threshold
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        # CHORUS-specific config
        config.cca_visual_path = args.cca_visual_path
        config.cca_text_path = getattr(args, 'cca_text_path', None)
        config.chorus_variant = getattr(args, 'chorus_variant', 'none')
        config.chorus_alpha_init = getattr(args, 'chorus_alpha_init', 1.0)
        config.chorus_beta_init = getattr(args, 'chorus_beta_init', 1.0)
        config.chorus_tau = getattr(args, 'chorus_tau', 0.3)
        # Per-stage disable flags
        config.disable_prestage = getattr(args, 'disable_prestage', False)
        config.disable_stage1_rho = getattr(args, 'disable_stage1_rho', False)
        config.disable_stage2_sigma = getattr(args, 'disable_stage2_sigma', False)
        config.disable_stage3_rho = getattr(args, 'disable_stage3_rho', False)
        config.disable_stage4_qk = getattr(args, 'disable_stage4_qk', False)
        config.disable_stage4_v = getattr(args, 'disable_stage4_v', False)
        config.disable_stage4_sigma = getattr(args, 'disable_stage4_sigma', False)
        model = CHORUS(config=config, num_classes=args.n_classes)
        print(f"\n[CHORUS] {model.visual_align}")

    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Expected CHORUS.")

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=10)

    print('\nInit Loaders...', end=' ')
    if _is_chorus(args):
        train_loader = get_split_loader_chorus(train_split, training=True,
                                                testing=args.testing,
                                                weighted=args.weighted_sample,
                                                num_workers=args.num_workers)
        val_loader = get_split_loader_chorus(val_split, testing=args.testing,
                                              num_workers=args.num_workers)
        test_loader = get_split_loader_chorus(test_split, testing=args.testing,
                                               num_workers=args.num_workers)
    else:
        train_loader = get_split_loader(train_split, training=True,
                                         testing=args.testing,
                                         weighted=args.weighted_sample,
                                         mode=args.mode)
        val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode)
        test_loader = get_split_loader(test_split, testing=args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(patience=args.patience, stop_epoch=args.stop_epoch, verbose=True) \
        if args.early_stopping else None
    print('Done!')

    scaler = GradScaler()

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer,
                   args.n_classes, writer, loss_fn, scaler)
        
        # Validation frequency logic
        val_every = getattr(args, 'val_every', 1)
        if (epoch + 1) % val_every == 0 or epoch == 0 or epoch == args.max_epochs - 1:
            stop = validate(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir, args)
            if stop:
                break

    if args.early_stopping:
        model.load_state_dict(torch.load(
            os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(),
                   os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader,
                                                args.n_classes, args)
    print(f'Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}, F1: {val_f1:.4f}')

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(
        args.mode, model, test_loader, args.n_classes, args)
    print(f'Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}, F1: {test_f1:.4f}')

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer:
            writer.add_scalar(f'final/test_class_{i}_acc', acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, each_class_acc, test_f1


def train_loop(args, epoch, model, loader, optimizer, n_classes,
               writer=None, loss_fn=None, scaler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss, train_error = 0., 0.
    print('\n')
    chorus = _is_chorus(args)

    for batch_idx, batch in enumerate(loader):
        x_s, x_l, label = _move_to_device(batch, device, chorus)
        
        with autocast('cuda'):
            _, Y_hat, loss = model(x_s, x_l, label)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        train_loss += loss.item()
        train_error += calculate_error(Y_hat, label)

    train_loss /= len(loader)
    train_error /= len(loader)
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_error: {train_error:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer:
            writer.add_scalar(f'train/class_{i}_acc', acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None,
             writer=None, loss_fn=None, results_dir=None, args=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss, val_error = 0., 0.
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    all_pred, all_label = [], []
    chorus = _is_chorus(args) if args is not None else False

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_s, x_l, label = _move_to_device(batch, device, chorus)
            with autocast('cuda'):
                Y_prob, Y_hat, loss = model(x_s, x_l, label)

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            val_error += calculate_error(Y_hat, label)
            all_pred.append(Y_hat.item())
            all_label.append(label.item())

    val_error /= len(loader)
    val_loss /= len(loader)
    val_f1 = f1_score(all_label, all_pred, average='macro')
    auc = roc_auc_score(labels, prob[:, 1]) if n_classes == 2 \
        else roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print(f'\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, '
          f'auc: {auc:.4f}, f1: {val_f1:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def summary(mode, model, loader, n_classes, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.
    all_pred, all_label = [], []
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    chorus = _is_chorus(args) if args is not None else False

    if mode == 'transformer':
        for batch_idx, batch in enumerate(loader):
            x_s, x_l, label = _move_to_device(batch, device, chorus)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                with autocast('cuda'):
                    Y_prob, Y_hat, loss = model(x_s, x_l, label)
            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            patient_results[slide_id] = {'slide_id': np.array(slide_id),
                                          'prob': probs, 'label': label.item()}
            test_error += calculate_error(Y_hat, label)
            all_pred.append(Y_hat.item())
            all_label.append(label.item())

        test_error /= len(loader)
        test_f1 = f1_score(all_label, all_pred, average='macro')
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=list(range(n_classes)))
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx],
                                             all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc = np.nanmean(np.array(aucs))
        return patient_results, test_error, auc, acc_logger, test_f1
