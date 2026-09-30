import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import gc
import pandas as pd
from model.MPTSNet import Model
from data_provider import get_label_dict, get_data_and_label_from_ts_file, fill_out_with_Nan
from utils import eval_condition, eval_model, save_to_log, fft_main_periods_wo_duplicates


# 修复设备配置 - 智能选择可用GPU并支持多GPU
def get_available_devices():
    if not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        return [torch.device('cpu')], False

    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个 GPU")

    # 优先级顺序：cuda:3 -> cuda:0 -> cuda:1 -> cuda:2 -> ...
    preferred_devices = [3, 0, 1, 2] + list(range(4, gpu_count))

    available_devices = []
    for device_id in preferred_devices:
        if device_id < gpu_count:
            try:
                # 测试设备是否可用
                test_tensor = torch.tensor([1.0]).cuda(device_id)
                device = torch.device(f'cuda:{device_id}')
                available_devices.append(device)
                print(f"GPU {device_id} 可用: {torch.cuda.get_device_name(device_id)}")
                del test_tensor  # 清理测试张量
            except Exception as e:
                print(f"GPU {device_id} 不可用: {e}")
                continue

    if len(available_devices) == 0:
        print("所有 GPU 都不可用，使用 CPU")
        return [torch.device('cpu')], False

    # 选择前两个可用的GPU进行多GPU训练
    if len(available_devices) >= 2:
        selected_devices = available_devices[:2]
        print(f"使用多GPU训练: {[str(d) for d in selected_devices]}")
        return selected_devices, True
    else:
        print(f"只有一个GPU可用，使用单GPU训练: {available_devices[0]}")
        return available_devices, False


devices, use_multi_gpu = get_available_devices()
primary_device = devices[0]
print(f"Primary device: {primary_device}")
if use_multi_gpu:
    print(f"Multi-GPU training enabled with devices: {[str(d) for d in devices]}")


def load_imbalanced_dataset(dataset_path, dataset_name, imbalance_ratio):
    """
    Load imbalanced training dataset with validation set - 适配.pt文件格式
    """
    # 构建文件路径
    train_file = f'{dataset_path}/train_LT{imbalance_ratio}.pt'
    val_file = f'{dataset_path}/val.pt'
    test_file = f'{dataset_path}/test.pt'
    original_train_file = f'{dataset_path}/train.pt'

    print(f"[INFO] Loading files:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"  Test: {test_file}")
    print(f"  Original: {original_train_file}")

    # 首先检查文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # 从原始训练文件获取标签字典（用于保持一致性）
    label_dict = {}
    if os.path.exists(original_train_file):
        try:
            label_dict = get_label_dict(original_train_file)
            print(f"[INFO] Loaded label dictionary with {len(label_dict)} classes: {label_dict}")
        except Exception as e:
            print(f"[WARNING] Could not load label dict from {original_train_file}: {e}")
            print("[INFO] Will infer labels from data files")

    try:
        # 加载不平衡训练数据
        print(f"[INFO] Loading training data from {train_file}")
        X_train, y_train = get_data_and_label_from_ts_file(train_file, label_dict)
        print(f"[INFO] Training data loaded: X shape={X_train.shape}, y shape={y_train.shape}")

        # 加载验证数据
        print(f"[INFO] Loading validation data from {val_file}")
        X_val, y_val = get_data_and_label_from_ts_file(val_file, label_dict)
        print(f"[INFO] Validation data loaded: X shape={X_val.shape}, y shape={y_val.shape}")

        # 加载测试数据
        print(f"[INFO] Loading test data from {test_file}")
        X_test, y_test = get_data_and_label_from_ts_file(test_file, label_dict)
        print(f"[INFO] Test data loaded: X shape={X_test.shape}, y shape={y_test.shape}")

        # 验证数据一致性
        print(f"[INFO] Data consistency check:")
        print(f"  Train labels range: {y_train.min()} - {y_train.max()}")
        print(f"  Val labels range: {y_val.min()} - {y_val.max()}")
        print(f"  Test labels range: {y_test.min()} - {y_test.max()}")

        # 检查数据形状一致性
        if len(X_train.shape) != len(X_val.shape) or len(X_train.shape) != len(X_test.shape):
            print("[WARNING] Data shapes are inconsistent across splits")

        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        raise


def calculate_mcc(conf_matrix):
    """
    计算多分类MCC (Matthews Correlation Coefficient)
    """
    try:
        from sklearn.metrics import matthews_corrcoef
        # 从混淆矩阵重构真实标签和预测标签
        y_true = []
        y_pred = []

        for true_class in range(conf_matrix.shape[0]):
            for pred_class in range(conf_matrix.shape[1]):
                count = conf_matrix[true_class, pred_class]
                y_true.extend([true_class] * count)
                y_pred.extend([pred_class] * count)

        mcc = matthews_corrcoef(y_true, y_pred)
        return mcc
    except Exception as e:
        print(f"Warning: Could not calculate MCC: {e}")
        return 0.0


def calculate_per_class_accuracy(y_true, y_pred, num_classes):
    """
    计算每个类别的准确率
    """
    per_class_acc = {}
    for class_id in range(num_classes):
        # 找到真实标签为该类别的样本
        class_mask = (y_true == class_id)
        if class_mask.sum() > 0:  # 确保该类别有样本
            # 计算该类别的准确率
            class_correct = (y_pred[class_mask] == class_id).sum()
            class_total = class_mask.sum()
            accuracy = class_correct / class_total
            per_class_acc[class_id] = accuracy
        else:
            per_class_acc[class_id] = 0.0  # 如果没有样本，准确率为0

    return per_class_acc


def train_single_run(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, imbalance_ratio, run_id,
                     result_folder, devices, use_multi_gpu):
    """
    单次训练运行 - 支持多GPU，优化显存使用
    """
    print(f'\n[INFO] Starting Run {run_id} for {dataset_name}_imbalance_{imbalance_ratio}')

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    primary_device = devices[0]

    num_channels = X_train.shape[1]
    embed_dim = max(min(num_channels * 4, 256), 64)
    embed_dim_t = max(min(embed_dim * 4, 512), 256)
    seq_length = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    # 转换为PyTorch张量但保持在CPU上进行预处理
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Z-norm标准化（在CPU上进行）
    if dataset_name in ['HAR', 'Epilepsy', 'ISRUC']:
        print('[INFO] Applying Z-norm for stable convergence')
        mean = X_train_tensor.mean(dim=(0, 2), keepdim=True)
        std = X_train_tensor.std(dim=(0, 2), keepdim=True)

        X_train_tensor = (X_train_tensor - mean) / (std + 1e-5)
        X_val_tensor = (X_val_tensor - mean) / (std + 1e-5)
        X_test_tensor = (X_test_tensor - mean) / (std + 1e-5)

    # 处理NaN值（在CPU上进行）
    X_train_tensor[torch.isnan(X_train_tensor)] = 0
    X_val_tensor[torch.isnan(X_val_tensor)] = 0
    X_test_tensor[torch.isnan(X_test_tensor)] = 0

    # 标签张量（保持在CPU，由DataLoader按需移动）
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)

    # 计算FFT周期（在CPU上进行）
    X_train_fft = X_train_tensor.permute(0, 2, 1).numpy()
    periods = fft_main_periods_wo_duplicates(X_train_fft, 5, f'{dataset_name}_imbalance_{imbalance_ratio}_run{run_id}')

    # 创建数据加载器（数据保持在CPU，由DataLoader按需移动到GPU）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)

    # 创建模型
    model = Model(periods=periods, flag=False, num_channels=num_channels, seq_length=seq_length,
                  num_classes=num_classes, embed_dim=embed_dim,
                  embed_dim_t=embed_dim_t, num_heads=4, ff_dim=256, num_layers=1).to(primary_device)

    # 如果使用多GPU，包装模型
    if use_multi_gpu and len(devices) > 1:
        device_ids = [d.index for d in devices if d.type == 'cuda']
        if len(device_ids) > 1:
            print(f"[INFO] Using DataParallel with devices: {device_ids}")
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")

    # 训练设置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=0.0001)

    best_val_acc = 0
    patience = 5
    cnt = 0
    model_save_path = f'{result_folder}best_model_run{run_id}'

    # 训练循环
    model.train()
    print(f'[INFO] Training started for Run {run_id}...')

    for epoch in range(100):
        running_loss = 0.0
        batch_count = 0

        for sample in train_loader:
            # 将数据移动到GPU（按需移动，不提前占用显存）
            data, target = sample[0].to(primary_device, non_blocking=True), sample[1].to(primary_device,
                                                                                         non_blocking=True)

            optimizer.zero_grad()
            y_predict = model(data)
            loss = criterion(y_predict, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count
        scheduler.step(avg_loss)

        if eval_condition(epoch, 1):
            model.eval()
            acc_train = eval_model(model, train_loader)
            acc_val = eval_model(model, val_loader)
            model.train()

            print(
                f'Run {run_id} - Epoch {epoch}: train_acc={acc_train:.4f}, val_acc={acc_val:.4f}, loss={avg_loss:.4f}')

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                # 保存模型时需要处理DataParallel包装
                if use_multi_gpu and isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
                print(f'Run {run_id} - New best model saved with val_acc={acc_val:.4f}')
                cnt = 0
            else:
                cnt += 1
                if cnt >= patience:
                    print(f'Run {run_id} - Early stopping after {epoch + 1} epochs')
                    break

    # 测试最佳模型
    print(f'[INFO] Testing best model for Run {run_id}...')

    # 加载模型时需要处理DataParallel包装
    if use_multi_gpu and isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(model_save_path))
    else:
        model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # 获取预测结果
    from utils import get_confmat_metrics
    import torch.nn.functional as F

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(primary_device, non_blocking=True)
            target = target.to(primary_device, non_blocking=True)
            output = model(data.float())
            pred = output.data.max(1, keepdim=False)[1]
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算指标
    final_test_acc = np.mean(all_preds == all_targets)

    from sklearn.metrics import confusion_matrix, f1_score
    conf_matrix = confusion_matrix(all_targets, all_preds)
    precision, recall, f1 = get_confmat_metrics(conf_matrix)

    # 计算macro F1
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    # 计算MCC
    mcc = calculate_mcc(conf_matrix)

    # 计算每个类别的准确率
    per_class_acc = calculate_per_class_accuracy(all_targets, all_preds, num_classes)

    print(f'Run {run_id} Results:')
    print(f'  Test Accuracy: {final_test_acc:.4f}')
    print(f'  Macro F1: {macro_f1:.4f}')
    print(f'  MCC: {mcc:.4f}')
    print(f'  Per-class Accuracy:')
    for class_id, acc in per_class_acc.items():
        print(f'    Class {class_id}: {acc:.4f}')

    # 保存每类准确率到文件
    per_class_results = pd.DataFrame([
        {'Class': class_id, 'Accuracy': acc}
        for class_id, acc in per_class_acc.items()
    ])
    per_class_save_path = f'{result_folder}per_class_accuracy_run{run_id}.csv'
    per_class_results.to_csv(per_class_save_path, index=False)
    print(f'[INFO] Per-class accuracy saved to: {per_class_save_path}')

    # 清理内存
    del model, optimizer, scheduler
    del X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
    del train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'run_id': run_id,
        'best_val_acc': best_val_acc,
        'test_acc': final_test_acc,
        'macro_f1': macro_f1,
        'mcc': mcc,
        'per_class_acc': per_class_acc
    }


if __name__ == '__main__':
    print('[INFO] Loading data...')

    # 设置PyTorch显存管理
    if torch.cuda.is_available():
        # 设置显存分配策略，避免碎片化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.cuda.empty_cache()

    # 选择数据集
    dataset_choice = 2  # 0 for HAR, 1 for Epilepsy, 2 for ISRUC
    dataset_names = ['HAR', 'Epilepsy', 'ISRUC','pm2.5']
    dataset_folder = dataset_names[dataset_choice]
    dataset_name = dataset_names[dataset_choice]

    dataset_path = f'./dataset/{dataset_folder}'
    #imbalance_ratios = [11]
    imbalance_ratios = [10, 50, 100]

    num_runs = 1  # 每个不平衡比例跑2轮

    Result_log_folder = f"./results/0803/MPTSNet_{dataset_name}/"
    if not os.path.exists(Result_log_folder):
        os.makedirs(Result_log_folder)

    # 保存所有结果
    all_results = []
    summary_results = []

    for imbalance_ratio in imbalance_ratios:
        print(f'\n{"=" * 80}')
        print(f'[INFO] Running {dataset_name} with imbalance ratio {imbalance_ratio}')
        print(f'{"=" * 80}')

        # 加载数据（所有轮次共用同一份数据）
        X_train, y_train, X_val, y_val, X_test, y_test = load_imbalanced_dataset(
            dataset_path, dataset_name, imbalance_ratio)

        print('[INFO] Data shapes:')
        print(f'  Train: X{X_train.shape}, y{y_train.shape}')
        print(f'  Val: X{X_val.shape}, y{y_val.shape}')
        print(f'  Test: X{X_test.shape}, y{y_test.shape}')

        # 处理序列长度不匹配
        max_length = max(X_train.shape[-1], X_val.shape[-1], X_test.shape[-1])
        if X_train.shape[-1] != max_length:
            X_train = fill_out_with_Nan(X_train, max_length)
        if X_val.shape[-1] != max_length:
            X_val = fill_out_with_Nan(X_val, max_length)
        if X_test.shape[-1] != max_length:
            X_test = fill_out_with_Nan(X_test, max_length)

        # 创建结果文件夹
        result_folder = Result_log_folder + f'{dataset_name}_imbalance_{imbalance_ratio}/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 多轮训练
        run_results = []
        for run_id in range(1, num_runs + 1):
            run_result = train_single_run(
                X_train, y_train, X_val, y_val, X_test, y_test,
                dataset_name, imbalance_ratio, run_id, result_folder, devices, use_multi_gpu
            )
            run_results.append(run_result)
            all_results.append({
                'Dataset': dataset_name,
                'Imbalance_Ratio': imbalance_ratio,
                'Run': run_id,
                'Best_Val_Acc': round(run_result['best_val_acc'], 4),
                'Test_Acc': round(run_result['test_acc'], 4),
                'Macro_F1': round(run_result['macro_f1'], 4),
                'MCC': round(run_result['mcc'], 4)
            })

        # 计算平均结果
        avg_test_acc = np.mean([r['test_acc'] for r in run_results])
        avg_macro_f1 = np.mean([r['macro_f1'] for r in run_results])
        avg_mcc = np.mean([r['mcc'] for r in run_results])
        std_test_acc = np.std([r['test_acc'] for r in run_results])
        std_macro_f1 = np.std([r['macro_f1'] for r in run_results])
        std_mcc = np.std([r['mcc'] for r in run_results])

        print(f'\n[SUMMARY] {dataset_name}_imbalance_{imbalance_ratio} ({num_runs} runs):')
        print(f'  Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}')
        print(f'  Average Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}')
        print(f'  Average MCC: {avg_mcc:.4f} ± {std_mcc:.4f}')

        summary_results.append({
            'Dataset': dataset_name,
            'Imbalance_Ratio': imbalance_ratio,
            'Avg_Test_Acc': round(avg_test_acc, 4),
            'Std_Test_Acc': round(std_test_acc, 4),
            'Avg_Macro_F1': round(avg_macro_f1, 4),
            'Std_Macro_F1': round(std_macro_f1, 4),
            'Avg_MCC': round(avg_mcc, 4),
            'Std_MCC': round(std_mcc, 4)
        })

        # 清理内存
        del X_train, y_train, X_val, y_val, X_test, y_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 打印显存使用情况
        if torch.cuda.is_available():
            for i, device in enumerate(devices):
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated(device.index) / 1024 ** 3
                    reserved = torch.cuda.memory_reserved(device.index) / 1024 ** 3
                    print(f"[INFO] GPU {device.index} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # 保存详细结果
    df_all = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_results)

    all_results_path = Result_log_folder + f'{dataset_name}_detailed_results.csv'
    summary_results_path = Result_log_folder + f'{dataset_name}_summary_results.csv'

    df_all.to_csv(all_results_path, index=False)
    df_summary.to_csv(summary_results_path, index=False)

    print(f'\n{"=" * 80}')
    print('[FINAL SUMMARY]')
    print(f'{"=" * 80}')
    for _, row in df_summary.iterrows():
        print(f"{row['Dataset']}_IR{row['Imbalance_Ratio']}:")
        print(f"  Test Acc: {row['Avg_Test_Acc']:.4f} ± {row['Std_Test_Acc']:.4f}")
        print(f"  Macro F1: {row['Avg_Macro_F1']:.4f} ± {row['Std_Macro_F1']:.4f}")
        print(f"  MCC: {row['Avg_MCC']:.4f} ± {row['Std_MCC']:.4f}")
        print()

    print(f'[INFO] Detailed results saved to: {all_results_path}')
    print(f'[INFO] Summary results saved to: {summary_results_path}')
    print('[INFO] Training completed!')