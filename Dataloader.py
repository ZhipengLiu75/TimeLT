import os

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import pywt


class Load_Dataset(Dataset):
    def __init__(self, dataset, traindata):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if traindata:
            print(X_train.shape, y_train.shape, '---------11111--------')
            print(f'This dataset has {max(y_train) + 1} classes')
            unique_classes = torch.unique(y_train)
            for cls in unique_classes:
                count = (y_train == cls).sum().item()
                print(f"Class {int(cls.item())}: {count} samples")

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).to('cuda')
            self.y_data = torch.from_numpy(y_train).long().to('cuda')
        else:
            self.x_data = X_train.float().to('cuda')
            self.y_data = y_train.long().to('cuda')

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Load_Dataset_Aug(Dataset):
    def __init__(self, dataset, traindata, balance_factor=.5, end=.1):
        super(Load_Dataset_Aug, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        self.balance_factor = balance_factor

        if traindata:
            unique_classes = torch.unique(y_train)
            print('Before Augmentation')
            for cls in unique_classes:
                count = (y_train == cls).sum().item()
                print(f"Class {int(cls.item())}: {count} samples")
            # base_std=0.0001
            base_std = []
            for x in X_train:
                base_std.append(torch.std(x).item())
            base_std = np.mean(base_std)
            print(np.round(base_std, 5), '----------------')


            X_train, y_train = self.balanced_class_aware_jitter_augment(X_train, y_train, base_std=base_std, end=end)

            print('After Augmentation')
            unique_classes = torch.unique(y_train)
            for cls in unique_classes:
                count = (y_train == cls).sum().item()
                print(f"Class {int(cls.item())}: {count} samples")

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).to('cuda')
            self.y_data = torch.from_numpy(y_train).long().to('cuda')
        else:
            self.x_data = X_train.float().to('cuda')
            self.y_data = y_train.long().to('cuda')

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def balanced_class_aware_jitter_augment(self, X, y, base_std, end):
        unique, counts = torch.unique(y, return_counts=True)
        sorted_counts = counts[unique.argsort()]

        jitter_scales= (torch.sum(sorted_counts)-sorted_counts)/ torch.sum(sorted_counts)

        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        X_aug_list = []
        y_aug_list = []


        print(base_std,'=======')
        print(base_std*np.round(jitter_scales.numpy(),5))
        for i, cls in enumerate(sorted(class_counts.keys())):
            X_cls = X[y == cls]
            N_cls = len(X_cls)
            n_to_add = int((max_count - N_cls) * self.balance_factor)


            for x in X_cls:
                X_aug_list.append(x)
                y_aug_list.append(cls)
            # 添加 jitter 增强样本
            for _ in range(n_to_add):
                x = X_cls[np.random.randint(0, N_cls)]
                noise = np.random.normal(loc=0, scale=base_std * jitter_scales[i], size=x.shape).astype(np.float32)
                X_aug_list.append(x + noise)
                y_aug_list.append(cls)

        X_aug = torch.tensor(np.stack(X_aug_list), dtype=torch.float32)
        y_aug = torch.tensor(np.array(y_aug_list), dtype=torch.long)

        return X_aug, y_aug


def data_generator(args):
    if args.mode == 'stepped':
        train_dataset = torch.load(os.path.join(args.data_path, "Stepped_train_LT" + str(args.beta) + ".pt"))
    if args.mode == 'exp':
        train_dataset = torch.load(os.path.join(args.data_path, "train_LT" + str(args.beta) + ".pt"))
    if args.mode == 'linear':
        train_dataset = torch.load(os.path.join(args.data_path, "Linear_train_LT" + str(args.beta) + ".pt"))

    # train_dataset = torch.load(os.path.join(args.data_path, "train_LT10.pt"))
    val_dataset = torch.load(os.path.join(args.data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = Load_Dataset_Aug(train_dataset, True, args.balance_factor, args.end)  # True, args.balance_factor
    val_dataset = Load_Dataset_Aug(val_dataset, False)
    test_dataset = Load_Dataset_Aug(test_dataset, False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batchsize,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader
