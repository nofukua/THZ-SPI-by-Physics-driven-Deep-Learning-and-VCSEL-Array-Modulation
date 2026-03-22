"""
# File       : UNETtraincode1.py
# Time       ：2025/8/31
# Author     ：hjj
# version    ：python 3.9, torch 2.3.1, CUDA 12.1，GPU RTX3060-12G
#设置参数      ：batch_size，num_epochs，C_VALUE（pattern 数量）
# Description：两者一起练
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import scipy.io as sio
import glob
from datetime import datetime
import sys
import re
import numpy as np

from UNETmodel_hjj1 import EnDecodingUNet  # 请确保路径正确

# =================== 配置 ===================

# 从命令行获取pattern数量
import argparse# <-- 自动设置 pattern 数量
parser = argparse.ArgumentParser()
parser.add_argument("--C", type=int, default=512, help="Pattern数量C")
parser.add_argument("--select_mode", type=int, default=4    , choices=[1, 2, 3, 4], help="Pattern初始化模式select_mode")
parser.add_argument("--random_seed", type=int, default=21, help="用于 select_mode=2 的随机种子")
args = parser.parse_args()
C_VALUE = args.C

#C_VALUE = 256  # <-- 手动设置 pattern 数量
BATCH_SIZE = 512
EPOCHS = 80
LR = 1e-2
Patience = 40  # 早停耐心值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 数据加载（先加载用于获取H,W） ===================
# 修改为加载噪声和干净数据对
noisy_train_data_path = "data/train_EMNISTBalanced_32x32_noise2dB_mix_lightfield_gauss_FS5.pth"
clean_train_data_path = "data/train_EMNISTBalanced_32x32_mix_contrast_expand9_FS5.pth"
noisy_test_data_path = "data/test_EMNISTBalanced_32x32_noise2dB_mix_lightfield_gauss.pth"
clean_test_data_path = "data/test_EMNISTBalanced_32x32_mix_contrast_expand9.pth"

# 从噪声文件名中提取噪声水平
def extract_noise_level(filepath):
    """从文件名中提取噪声水平(dB)"""
    filename = os.path.basename(filepath)
    match = re.search(r'noise(\d+)dB', filename)
    return match.group(1) if match else "unknown"

noise_level = extract_noise_level(noisy_train_data_path)  # 提取噪声水平，如"35"

# 加载数据
noisy_train_data = torch.load(noisy_train_data_path)
clean_train_data = torch.load(clean_train_data_path)
noisy_test_data = torch.load(noisy_test_data_path)
clean_test_data = torch.load(clean_test_data_path)

# 获取图像尺寸 H, W
_, _, H, W = noisy_train_data.shape
print(f"图像尺寸: H={H}, W={W}, Pattern数量C={C_VALUE}, 噪声水平={noise_level}dB")

# 创建数据加载器（噪声输入，干净目标）
train_loader = DataLoader(
    TensorDataset(noisy_train_data, clean_train_data),
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(noisy_test_data, clean_test_data),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =================== 初始化模型 ===================
SELECT_MODE = args.select_mode
#model = EnDecodingUNet(H=H, W=W, C=C_VALUE, select_mode=SELECT_MODE).to(DEVICE)
model = EnDecodingUNet(
    H=H, W=W, C=C_VALUE,
    select_mode=SELECT_MODE,
    random_seed=args.random_seed  # ✅ 加上这行
).to(DEVICE)




# 自动创建日期时间文件夹（包含C值和噪声水平）
# 🔹提取数据集名称（如：Data_MNIST32）
dataset_folder_name = os.path.basename(os.path.dirname(noisy_train_data_path))
# 🔹生成时间戳
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 🔹获取当前脚本的文件名（不带扩展名）
script = os.path.splitext(os.path.basename(__file__))[0]  # e.g., 'traincode13'
script_name = f"{script}_results"                # traincode13_results ✅
# 🔹构建保存路径结构
subfolder_name = f"C{C_VALUE}_{dataset_folder_name}_noise{noise_level}dB_{current_time}"
RESULTS_ROOT = os.path.join("Results", dataset_folder_name)
# ✅ 插入 script_name 文件夹作为中间层
SCRIPT_FOLDER = os.path.join(RESULTS_ROOT, script_name)
SAVE_ROOT = os.path.join(SCRIPT_FOLDER, subfolder_name)
# 🔹各路径
MODEL_SAVE_PATH = os.path.join(SAVE_ROOT, "Model")
PATTERN_SAVE_PATH = os.path.join(SAVE_ROOT, "pattern")
LOG_SAVE_PATH = os.path.join(SAVE_ROOT, "log")
# 🔹创建目录
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(PATTERN_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)

# 初始化日志记录
log_file = os.path.join(LOG_SAVE_PATH, "training_log.txt")
log_metrics = {
    'epoch': [],
    'train_loss': [],
    'test_loss': [],
    'ssim': [],
    'psnr': []
}


# 重定向标准输出到日志文件
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        self.terminal.flush()


sys.stdout = Logger(log_file)

# =================== 模型 & 优化器 ===================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 添加预热调度器
WARMUP_EPOCHS = 5  # 预热轮数

def warmup_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    else:
        return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # 监控SSIM（越大越好）
    factor=0.5,  # 学习率衰减因子
    patience=7,  # 连续3个epoch无改善则衰减学习率
    threshold=0.0005,  # 视为改善的最小变化量
    min_lr=1e-4  # 最小学习率
)

best_ssim = -1.0
best_model_path = None
best_epoch = -1


# =================== EarlyStopping类 ===================
class EarlyStopping:
    """提前停止类，用于监控指标并决定是否停止训练"""
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.lr_reset_done = False  # 新增学习率重置标记

    def __call__(self, val_ssim, val_psnr, model):
        score = val_ssim

        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            return False

        ssim_drop = self.best_score - score

        if ssim_drop <= 0.005 and val_psnr > log_metrics['psnr'][-1] + 0.01:
            psnr_gain = val_psnr - log_metrics['psnr'][-1]
            print(f"🌟 尽管SSIM略降({ssim_drop:.4f})，但PSNR补偿 {psnr_gain:.4f}，认为是性能提升")
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            print(
                f"EarlyStopping: {self.counter}/{self.patience} (当前SSIM: {score:.4f}, 最佳SSIM: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                    print(f"恢复最佳权重 (SSIM: {self.best_score:.4f})")
                return True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
            print(f"EarlyStopping: SSIM指标改善到 {score:.4f}，重置计数")

        return False


early_stopping = EarlyStopping(
    patience=Patience,
    min_delta=0.0005,
    restore_best_weights=True
)


# =================== 工具函数 ===================
def compute_metrics(output, target):
    """计算重建图像与干净目标之间的指标"""
    output_np = output.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    ssim = ssim_metric(target_np, output_np, data_range=1.0)
    psnr = psnr_metric(target_np, output_np, data_range=1.0)
    return ssim, psnr

def save_patterns(patterns, epoch, ssim, psnr, folder, is_best=False):
    """保存patterns为pth和mat格式"""
    prefix = "best_" if is_best else ""
    pattern_name = f"{prefix}patterns_epoch{epoch}_noise_ssim{ssim:.4f}_psnr{psnr:.2f}.pth"
    torch.save(patterns.cpu(), os.path.join(folder, pattern_name))

    mat_name = f"{prefix}patterns_epoch{epoch}_noise_ssim{ssim:.4f}_psnr{psnr:.2f}.mat"
    sio.savemat(os.path.join(folder, mat_name),
                {"patterns": patterns.detach().cpu().numpy()})

    print(f"✅ 保存patterns到 {mat_name} 和 {pattern_name}")

def plot_training_curves(metrics, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(metrics['epoch'], metrics['ssim'])
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity (SSIM)')

    plt.subplot(1, 3, 3)
    plt.plot(metrics['epoch'], metrics['psnr'])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Peak Signal-to-Noise Ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def append_script_to_log(log_file: str, script_path: str = __file__):
    """将当前训练脚本源码写入日志文件"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code_text = f.read()
        with open(log_file, 'a', encoding='utf-8') as logf:
            logf.write("\n\n" + "⇩"*80 + "\n")
            logf.write("# 当前训练脚本完整代码：\n\n")
            logf.write(code_text)
            logf.write("\n" + "⇧"*80 + "\n")
        print(f"✅ 当前脚本代码已追加写入日志文件 {log_file}")
    except Exception as e:
        print(f"⚠️ 写入脚本代码到日志时出错: {e}")
def append_model_to_log(log_file: str, script_path: str = __file__):
    """自动查找并写入模型定义文件源码到日志"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            traincode_text = f.read()

        # 查找形如 from model_xxx import EnDecodingUNet 的导入语句
        model_import_match = re.search(r'from\s+(model_\w+)\s+import\s+EnDecodingUNet', traincode_text)

        if model_import_match:
            model_module_name = model_import_match.group(1)

            # 获取模型文件路径
            spec = importlib.util.find_spec(model_module_name)
            if spec and spec.origin and spec.origin.endswith('.py'):
                model_path = spec.origin
                with open(model_path, 'r', encoding='utf-8') as f:
                    model_code = f.read()

                with open(log_file, 'a', encoding='utf-8') as logf:
                    logf.write("\n\n" + "⇩" * 80 + "\n")
                    logf.write(f"# 模型定义文件 ({os.path.basename(model_path)}) 源码：\n\n")
                    logf.write(model_code)
                    logf.write("\n" + "⇧" * 80 + "\n")

                print(f"✅ 模型代码 {model_path} 已自动写入日志文件 {log_file}")
            else:
                print(f"⚠️ 未能找到模型模块 {model_module_name} 的源码文件路径")
        else:
            print("⚠️ 未能从训练脚本中识别模型导入语句（格式应为: from model_xxx import EnDecodingUNet）")
    except Exception as e:
        print(f"⚠️ 写入模型定义代码到日志时出错: {e}")


# =================== 开始训练 ===================
print("=" * 80)
print("🚀 开始训练 EnDecodingUNet 模型 (噪声输入-干净目标)")
print(f"🕒 开始时间       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 数据集         : {dataset_folder_name}")
print(f"🧊 图像尺寸       : H={H}, W={W}")
print(f"📈 Pattern 数量   : C={C_VALUE}")
print(f"🔉 噪声水平       : {noise_level}dB")
print(f"🧠 模型结构       : EnDecodingUNet")
print(f"⚙️ 参数配置:")
print(f"   ▸ Epochs        = {EPOCHS}")
print(f"   ▸ Batch Size    = {BATCH_SIZE}")
print(f"   ▸ Learning Rate = {LR}")
print(f"   ▸ Device        = {DEVICE}")
print(f"   ▸ Select Mode   = {SELECT_MODE}  # Pattern初始化方式 (1=ZigzagHadamard, 2=随机Hadamard, 3=zigzag等间距, 4=纯随机)")
print(f"   ▸ Random Seed   = {args.random_seed}")
print(f"📁 结果保存路径   : {RESULTS_ROOT}")
print(f"📁 模型保存路径   : {MODEL_SAVE_PATH}")
print(f"📁 Pattern保存路径: {PATTERN_SAVE_PATH}")
print(f"📝 日志文件       : {log_file}")
print(f"📉 使用EarlyStopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
print("=" * 80 + "\n")


# 训练前保存初始patterns
with torch.no_grad():
    initial_patterns = model.pattern_generator().detach().cpu()
    save_patterns(initial_patterns, 0, 0.0, 0.0, PATTERN_SAVE_PATH)

try:
    best_psnr = -1.0
    overall_start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        print(f"\n🟢 正在训练第 {epoch_num} / {EPOCHS} 轮...")

        # =================== 训练阶段 ===================
        model.train()
        start_time = time.time()
        total_loss = 0

        for noisy_imgs, clean_imgs in train_loader:  # 修改为解包噪声输入和干净目标
            noisy_imgs = noisy_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            # 前向传播（输入噪声图像）
            patterns_bin, _, _, _, _, recon = model(noisy_imgs)

            # 损失计算（与干净目标比较）
            loss = criterion(recon, clean_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time

        # =================== 验证阶段 ===================
        model.eval()
        test_loss = 0.0
        ssim_list = []
        psnr_list = []

        with torch.no_grad():
            for noisy_imgs, clean_imgs in test_loader:  # 修改为解包噪声输入和干净目标
                noisy_imgs = noisy_imgs.to(DEVICE)
                clean_imgs = clean_imgs.to(DEVICE)

                patterns_bin, _, _, _, _, recon = model(noisy_imgs)
                loss = criterion(recon, clean_imgs)  # 与干净目标比较
                test_loss += loss.item()

                for i in range(clean_imgs.size(0)):
                    # 计算重建结果与干净目标的指标
                    ssim_val, psnr_val = compute_metrics(recon[i], clean_imgs[i])
                    ssim_list.append(ssim_val)
                    psnr_list.append(psnr_val)

        avg_test_loss = test_loss / len(test_loader)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_psnr = sum(psnr_list) / len(psnr_list)

        # 记录训练指标
        log_metrics['epoch'].append(epoch_num)
        log_metrics['train_loss'].append(avg_train_loss)
        log_metrics['test_loss'].append(avg_test_loss)
        log_metrics['ssim'].append(avg_ssim)
        log_metrics['psnr'].append(avg_psnr)

        current_lr = optimizer.param_groups[0]['lr']
        total_elapsed = time.time() - overall_start_time
        hours, rem = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"⏱️ 耗时: {elapsed:.2f}s | 累计总训练耗时: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")
        print(f"📉 Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}| 当前学习率: {current_lr:.2e}")
        print(f"🏆 SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f} dB")


        # =================== 学习率调度 ===================
        old_lr = optimizer.param_groups[0]['lr']

        # 应用学习率调度器
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            print(f"🔥 Warmup阶段: 学习率已更新为 {optimizer.param_groups[0]['lr']:.2e}")
        else:
            # 检查是否需要重置学习率
            if not hasattr(early_stopping, 'lr_reset_done') and \
                    early_stopping.counter >= int(0.7 * early_stopping.patience):
                # 重置学习率为初始值的一半
                reset_lr = LR * 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = reset_lr
                early_stopping.lr_reset_done = True  # 标记已重置
                print(
                    f"⏫ 早停计数器已达{early_stopping.counter}/{early_stopping.patience}（70%），重置学习率为初始值的一半: {reset_lr:.2e}")
            else:
                # 正常调度
                scheduler.step(avg_ssim)
                print(f"📉 Scheduler更新: 学习率为 {optimizer.param_groups[0]['lr']:.2e}")

        new_lr = optimizer.param_groups[0]['lr']



        if new_lr != old_lr and not (early_stopping.counter >= int(0.7 * early_stopping.patience)):
            print(f"学习率从 {old_lr:.2e} 调整到 {new_lr:.2e}")

        if new_lr != old_lr:
            # 如果是由于早停70%导致的调整，已经在上面的逻辑中打印过特定信息
            if not (early_stopping.counter >= int(0.7 * early_stopping.patience)):
                print(f"学习率从 {old_lr:.2e} 调整到 {new_lr:.2e}")

        # =================== 保存模型和pattern ===================
        is_best = False
        ssim_improved = avg_ssim > best_ssim
        ssim_drop = best_ssim - avg_ssim if best_ssim != -1.0 else 0
        psnr_gain = avg_psnr - (log_metrics['psnr'][-1] if log_metrics['psnr'] else 0)

        if ssim_improved or (ssim_drop <= 0.005 and psnr_gain >= 0.01):
            best_ssim = avg_ssim
            best_epoch = epoch_num
            is_best = True
            model_name = f"best_model_epoch{best_epoch}_noise_ssim{best_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
            best_model_path = os.path.join(MODEL_SAVE_PATH, model_name)
            print(f"🌟 当前模型被判定为最佳模型: {model_name}（ssim_drop={ssim_drop:.4f}, psnr_gain={psnr_gain:.2f}）")
        else:
            model_name = f"model_epoch{epoch_num}_noise_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
            best_model_path = os.path.join(MODEL_SAVE_PATH, model_name)
            print(f"💾 保存当前模型: {model_name}")

        # 保存模型
        model_path = os.path.join(MODEL_SAVE_PATH, model_name)
        torch.save({
            'model': model.state_dict(),
            'pattern_shape': (model.C, model.H, model.W)
        }, model_path)

        # 更新最佳模型路径
        if is_best:
            best_model_path = model_path
            print(f"💾 保存模型: {model_name} (标记为最佳)")
        else:
            print(f"💾 保存模型: {model_name}")

        # 保存patterns
        # 每一轮都保存pattern，仅最佳时加best_前缀
        save_patterns(patterns_bin, epoch_num, avg_ssim, avg_psnr, PATTERN_SAVE_PATH, is_best=is_best)

        # =================== EarlyStopping检查 ===================
        if early_stopping(avg_ssim, avg_psnr, model):
            print(f"⛔️ EarlyStopping触发！在Epoch {epoch_num}停止训练")
            break

except KeyboardInterrupt:
    print("\n⛔️ 手动终止训练，正在加载最佳模型进行可视化...")
    if best_ssim > -1:
        print("💾 保存当前patterns...")
        model.eval()
        with torch.no_grad():
            patterns_bin = model.pattern_generator().detach().cpu()
            save_patterns(patterns_bin, epoch_num, avg_ssim, avg_psnr, PATTERN_SAVE_PATH, is_best=is_best)

    # 确保手动终止时维度一致
    if len(log_metrics['epoch']) > len(log_metrics['train_loss']):
        log_metrics['epoch'].pop()

# 保存训练曲线
plot_training_curves(log_metrics, LOG_SAVE_PATH)

# =================== 加载最佳模型 & 可视化 ===================
if best_model_path:
    print(f"\n✅ 加载最佳模型: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 加载patterns
    matched_patterns = glob.glob(os.path.join(
        PATTERN_SAVE_PATH,
        f"patterns_epoch{best_epoch}_noise_ssim{best_ssim:.4f}_psnr*.pth"
    ))

    if matched_patterns:
        print(f"✅ 加载最佳patterns: {os.path.basename(matched_patterns[0])}")
        patterns = torch.load(matched_patterns[0])
    else:
        print("⚠️ 未找到最佳patterns文件，重新生成...")
        with torch.no_grad():
            patterns = model.pattern_generator().detach().cpu()

    # 可视化重建结果
    with torch.no_grad():
        # 获取测试集前3个样本（噪声输入和干净目标）
        noisy_imgs, clean_imgs = next(iter(test_loader))
        noisy_imgs = noisy_imgs[:3].to(DEVICE)
        clean_imgs = clean_imgs[:3].to(DEVICE)

        # 重建噪声图像
        _, _, DGI_norm, _, enhanced_norm, _ = model(noisy_imgs)

    # =================== 显示噪声输入、重建结果和干净目标对比 ===================
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(f"Best Model: Epoch {best_epoch}, SSIM {best_ssim:.4f}, PSNR {avg_psnr:.2f}", fontsize=12)

    for i in range(3):
        # 噪声输入
        axes[i, 0].imshow(noisy_imgs[i].squeeze().cpu(), cmap='gray')
        axes[i, 0].set_title(f"Noisy Input #{i + 1}")
        axes[i, 0].axis('off')

        # U-Net增强重建
        axes[i, 1].imshow(enhanced_norm[i].squeeze().cpu(), cmap='gray')
        axes[i, 1].set_title(f"Reconstruction #{i + 1}")
        axes[i, 1].axis('off')

        # 干净目标
        axes[i, 2].imshow(clean_imgs[i].squeeze().cpu(), cmap='gray')
        axes[i, 2].set_title(f"Clean Target #{i + 1}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, "best_reconstruction_comparison.png"))
    plt.close()

    # =================== 可视化Patterns (12个示例) ===================
    print(f"\n🖼️ 显示最佳模型的patterns (C={patterns.shape[0]})")
    num_to_show = min(12, patterns.shape[0])
    cols = 3
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle(f"Best Patterns: Epoch {best_epoch}", fontsize=14)
    for i in range(num_to_show):
        if rows > 1:
            ax = axes[i // cols, i % cols]
        else:
            ax = axes[i % cols]
        ax.imshow(patterns[i].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Pattern #{i + 1}")
        ax.axis('off')
    for i in range(num_to_show, rows * cols):
        if rows > 1:
            axes[i // cols, i % cols].axis('off')
        else:
            axes[i % cols].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PATTERN_SAVE_PATH, "best_patterns_preview.png"))
    plt.close()

else:
    print("⚠️ 没有保存任何模型，请检查训练是否正常执行。")
    exit()

# 恢复标准输出
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

print(f"\n{'=' * 80}")
print(f"🏁 训练完成! 共训练 {len(log_metrics['epoch'])} 轮（共 {EPOCHS} 轮）")
print(f"🖼️ 最佳模型: epoch={best_epoch}, SSIM={best_ssim:.4f}, PSNR={avg_psnr:.2f} dB")
print(f"📂 所有结果保存在: {RESULTS_ROOT}")
print(f"💾 模型路径: {MODEL_SAVE_PATH}")
print(f"🖼️ Patterns路径: {PATTERN_SAVE_PATH}")
print(f"📝 训练日志: {log_file}")
print(f"📈 训练曲线图: {os.path.join(LOG_SAVE_PATH, 'training_curves.png')}")
print("=" * 80)
import importlib.util


# === 写入当前训练脚本源码和模型定义源码到日志 ===
append_script_to_log(log_file)
append_model_to_log(log_file)

