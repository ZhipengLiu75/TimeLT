import argparse
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import optim


from Dataloader import Load_Dataset, Load_Dataset_Aug, data_generator
from models import *

from util import BoundaryLoss
from typing import List

parser = argparse.ArgumentParser()

parser.add_argument('--epoches', default=100, type=int)
parser.add_argument('--data_path', default='data/HAR/', type=str)
parser.add_argument('--seq_len', default=128, type=int)
parser.add_argument('--indim', default=9, type=int)
parser.add_argument('--classes', default=6, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--alpha1', default=.1, type=float)
parser.add_argument('--alpha2', default=.1, type=float)
parser.add_argument('--IR', default=10, type=float)

parser.add_argument('--down_sampling_layers', default=2, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--temporal_size', default=128, type=int)
parser.add_argument('--balance_factor', default=.5, type=int)
parser.add_argument('--end', default=0.1, type=float)


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = Model(args).to('cuda')
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
    bestmodel.eval()
    for data, label in test_loader:
        out, features = bestmodel(data)
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
    boundaryloss = BoundaryLoss()

    for epoch in range(1, args.epoches + 1):
        train_losses, val_losses, test_losses = [], [], []
        model.train()
        for data, label in train_loader:
            optimizer.zero_grad()
            out, features = model(data)
            loss = PredLossFun(out, label)
            lb, ld = boundaryloss(features, label)
            loss += args.alpha1 * lb + args.alpha2 * ld
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for data, label in val_loader:
                out, features = model(data)
                loss = PredLossFun(out, label)
                val_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            for data, label in test_loader:
                out, features = model(data)
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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = data_generator(args)
    model = Model(args).float().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True, dump=False)

    Trainer(args, model, optimizer, train_loader, val_loader, test_loader, early_stopping, scheduler)

    acc, f1, mcc, X = Test(test_loader, early_stopping.best_model, True)
    return acc, f1, mcc, X


if __name__ == '__main__':
    args = parser.parse_args()
    acc, f1, mcc, x = main(args)
    print(acc, f1, mcc)
