import argparse
import time
import gc
import random

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
# import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

from net import GNNStack
from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='Datasets_Name')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')


class EarlyStopping:
    def __init__(self, args, seq_length, num_nodes, num_classes, patience=15, verbose=False, dump=False, ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        # self.best_model = Model(args).to('cuda')

        self.best_model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers,
                                   groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                                   in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                                   seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes).float()

        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())

        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0


def Test(test_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval().to('cuda')
    for data, label in test_loader:
        data = data.to('cuda')
        label = label.to('cuda')
        label = label.long()
        out,embedding = bestmodel(data)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)

        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    X = None
    if final == True:
        num_classes = torch.max(labels).item() + 1
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)

        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()

        per_class_acc = correct / total.clamp(min=1)  # 防止除以0
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item() * 100)
            print(f"Class {i} Accuracy: {per_class_acc[i].item() * 100:.2f}%", '总数是:', int(total[i].item()))

    acc = accuracy_score(predicts, labels) * 100
    f1 = f1_score(predicts, labels, average="macro") * 100
    mcc = matthews_corrcoef(predicts, labels)
    printtext = "ACC:{:.2f}".format(acc) + '% ' + 'F1:{:.2f}'.format(
        f1) + '% ' + 'MCC:{:.4f}'.format(
        mcc)
    print(printtext)
    return acc, f1, mcc, X


def main(args):
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date


    train_loader, val_loader, test_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(args)

    # training model from net.py
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers,
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                     seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes)

    early_stopping = EarlyStopping(args, seq_length, num_nodes, num_classes, patience=5, verbose=True, dump=False)

    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gc.collect()
        torch.cuda.empty_cache()
        model = model.cuda(args.gpu)
    else:
        print("Error!")

    model=model.float()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                              patience=3, verbose=True)

    # validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, args.epochs + 1):
        train_losses, val_losses, test_losses = [], [], []
        model.train()
        for count, (data, label) in enumerate(train_loader):
            label=label.long()
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            # compute output
            out,embedding = model(data)

            loss = criterion(out, label)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for data, label in val_loader:
                label = label.long()
                data = data.to(device)
                label = label.to(device)
                out,embedding = model(data)
                loss = criterion(out, label)
                val_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            for data, label in test_loader:
                label = label.long()
                data = data.to(device)
                label = label.to(device)
                out,embedding = model(data)
                loss = criterion(out, label)
                test_losses.append(loss.item())
        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, test_loss:{3:.5f}'.format(epoch,
                                                                                           np.mean(train_losses),
                                                                                           np.mean(val_losses),
                                                                                           np.mean(test_losses)))

        lr_scheduler.step(np.mean(val_losses))
        early_stopping(np.mean(val_losses), model, epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
            break
        if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
            break

        Test(test_loader, early_stopping.best_model, False)


    acc, f1, mcc, X = Test(test_loader, early_stopping.best_model, final=True)
    return acc, f1, mcc, X


def trimmed_mean_variance(data):
    trimmed = sorted(data)[1:-1]  # 去掉最大最小值
    mean = np.mean(data)
    variance = np.std(data)  # 默认是样本方差
    return mean, variance


if __name__ == '__main__':
    args = parser.parse_args()
    args.kern_size = [int(l) for l in args.kern_size.split(",")]
    for beta in [100]:
        args.beta = beta
        start = time.time()
        ACC = []
        F1 = []
        MCC = []
        X = []
        for i in range(10):
            acc, f1, mcc, x = main(args)
            print(acc, f1, mcc)
            ACC.append(acc)
            F1.append(f1)
            MCC.append(mcc)
            X.append(x)
            torch.cuda.empty_cache()

        X = np.stack(X, axis=0)
        print(X)
        print(ACC)
        print(F1)
        print(MCC)
        print('\033[31m===================', beta, '===============================\033[0m')
        accmean, accstd = trimmed_mean_variance(ACC)
        f1mean, f1std = trimmed_mean_variance(F1)
        mccmean, mccstd = trimmed_mean_variance(MCC)
        print('\033[31m ACC:', accmean, accstd, '\033[0m')
        print('\033[31m F1:', f1mean, f1std, '\033[0m')
        print('\033[31m MCC:', mccmean, mccstd, '\033[0m')
        print('\033[31m', np.mean(X, axis=0), '\033[0m')
        print('\033[31m', np.std(X, axis=0), '\033[0m')
        print(time.time() - start, '秒-------------')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')
