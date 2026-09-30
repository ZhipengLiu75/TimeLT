import argparse
import time
import numpy as np
import torch
from PatchMLP import PatchMLP
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import optim, nn
import os
from Libs.Dataloader import Load_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data/HAR/', type=str)
parser.add_argument('--num_class', default=6, type=int)
parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
parser.add_argument('--IR', type=int, default=10, help='decoder input size')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epoches', default=100, type=int)
parser.add_argument('--model', type=str, default='PatchMLP')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
parser.add_argument('--moving_avg', type=int, default=13, help='window size of moving average')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False, ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        if args.model == 'PatchMLP':
            self.best_model = PatchMLP(args).float().to('cuda')
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


def data_generator(args):
    train_dataset = torch.load(os.path.join(args.data_path, "train_LT" + str(args.IR) + ".pt"))
    val_dataset = torch.load(os.path.join(args.data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, True)
    val_dataset = Load_Dataset(val_dataset, False)
    test_dataset = Load_Dataset(test_dataset, False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batchsize,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader


def Test(test_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval()
    for data, label in test_loader:
        out = bestmodel(data)
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

        per_class_acc = correct / total.clamp(min=1)
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


def Trainer(args, model, optimizer, train_loader, val_loader, test_loader, early_stopping, scheduler=None):
    PredLossFun = nn.CrossEntropyLoss()
    for epoch in range(1, args.epoches + 1):
        train_losses, val_losses, test_losses = [], [], []
        model.train()
        for data, label in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = PredLossFun(out, label)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for data, label in val_loader:
                out = model(data)
                loss = PredLossFun(out, label)
                val_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            for data, label in test_loader:
                out = model(data)
                loss = PredLossFun(out, label)
                test_losses.append(loss.item())
        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, test_loss:{3:.5f}'.format(epoch,
                                                                                           np.mean(train_losses),
                                                                                           np.mean(val_losses),
                                                                                           np.mean(test_losses)))

        scheduler.step(np.mean(val_losses))
        early_stopping(np.mean(val_losses), model, epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
            break
        if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
            break

        Test(test_loader, early_stopping.best_model, False)


def main(args):
    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = data_generator(args)
    model = None
    if args.model == 'PatchMLP':
        model = PatchMLP(args).float().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True, dump=False)

    Trainer(args, model, optimizer, train_loader, val_loader, test_loader, early_stopping, scheduler)

    acc, f1, mcc, X = Test(test_loader, early_stopping.best_model, final=True)
    return acc, f1, mcc, X


if __name__ == '__main__':
    args = parser.parse_args()
    acc, f1, mcc, x = main(args)
    print(acc, f1, mcc)
