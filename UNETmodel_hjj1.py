
"""
# File       : model_hjj1unet.py
# Time       ：2025/7/17
# Author     ：hjj
# version    ：python 3.9, torch 2.3.1, CUDA 12.1
#设置参数      ：
# Description：使用RowColPatternGenerator模块产生正交pattern
               在训练里面传入H, W, C三个重要参数
               哈达玛pattern用作初始化
               融合了model_hjj3和model_hjj4，通过select_mode1，2，3来选择，1是zigzag前C个pattern，2是随机哈达玛pattern，3是纯随机

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''


class RowColPatternGenerator(nn.Module):
    def __init__(self, H, W, C, select_mode=1, init_zigzag=None, random_seed=None):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.select_mode = select_mode
        self.random_seed = random_seed

        # 统一由init_zigzag参数决定是否初始化Hadamard，默认根据select_mode自动判断
        if init_zigzag is None:
            self.init_zigzag = (self.select_mode in [1, 2, 3])
        else:
            self.init_zigzag = init_zigzag

        self.row_vec = nn.Parameter(torch.rand(C, H))#连续均匀分布
        self.col_vec = nn.Parameter(torch.rand(C, W))

        if self.init_zigzag:
            row_init, col_init = self._init_from_zigzag_hadamard()
            with torch.no_grad():
                self.row_vec.copy_(row_init)
                self.col_vec.copy_(col_init)
            print(f"✅ 使用 Hadamard Zigzag 初始化 (mode={self.select_mode})")
        else:
            print(f"✅ 使用随机初始化 (mode={self.select_mode}, init_zigzag={self.init_zigzag})")




    def forward(self):
        # 原硬二值化 + STE
        row_bin = torch.round(torch.clamp(self.row_vec, 0, 1))
        row_vec_bin = self.row_vec + (row_bin - self.row_vec).detach()

        col_bin = torch.round(torch.clamp(self.col_vec, 0, 1))
        col_vec_bin = self.col_vec + (col_bin - self.col_vec).detach()

        row = row_vec_bin.unsqueeze(2)  # [C,H,1]
        col = col_vec_bin.unsqueeze(1)  # [C,1,W]
        pattern = row * (1 - col) + (1 - row) * col
        return pattern

    def _init_from_zigzag_hadamard(self):
        # 1. 生成 n×n Hadamard
        n = self.H * self.H
        Hfull = np.array([[1]])
        while Hfull.shape[0] < n:
            Hfull = np.vstack((np.hstack((Hfull, Hfull)),
                               np.hstack((Hfull, -Hfull))))

        # 2. Gray-code 排序
        def reverse_bits(x, bl):
            rv = 0
            for i in range(bl):
                if (x >> i) & 1:
                    rv |= 1 << (bl - 1 - i)
            return rv

        def gray_seq(n):
            return [i ^ (i >> 1) for i in range(n)]

        gray_map = {g: i for i, g in enumerate(gray_seq(n))}
        bl = (n - 1).bit_length()
        idx = sorted(range(n), key=lambda i: gray_map[reverse_bits(i, bl)])
        Hs = Hfull[idx, :]

        # 3. 拆成 H×H 小块
        p = self.H
        blocks = [[Hs[i * p:(i + 1) * p, j * p:(j + 1) * p]
                   for j in range(p)] for i in range(p)]

        # 4. Zigzag 顺序遍历所有块索引
        zz = []
        for s in range(2 * p - 1):
            if s % 2 == 0:
                x0 = 0 if s < p else s - p + 1
                y0 = s if s < p else p - 1
                while x0 < p and y0 >= 0:
                    zz.append((x0, y0))
                    x0 += 1
                    y0 -= 1
            else:
                y0 = 0 if s < p else s - p + 1
                x0 = s if s < p else p - 1
                while x0 >= 0 and y0 < p:
                    zz.append((x0, y0))
                    x0 -= 1
                    y0 += 1

        # 5. 根据模式选择 block 索引
        if self.select_mode == 1:
            # 顺序选择
            selected_blocks = zz[:self.C]
        elif self.select_mode == 2:
            # 随机选择
            rng = np.random.default_rng(self.random_seed)
            selected_blocks = rng.choice(zz, size=self.C, replace=False)
        elif self.select_mode == 3:
            # 等间距选择
            if self.C > len(zz):
                raise ValueError(f"C={self.C} 超出可选模式数量 {len(zz)}")
            step = len(zz) / self.C
            indices = [int(i * step) for i in range(self.C)]
            selected_blocks = [zz[i] for i in indices]
        else:
            raise ValueError("select_mode must be 1 (sequential), 2 (random), or 3 (even spacing)")

        # 6. 提取对应行列
        row_list, col_list = [], []
        for i, j in selected_blocks:
            b = blocks[i][j]  # shape [H,H]
            rbin = ((b[0, :] + 1) // 2).astype(np.int32)  # -1→0, +1→1
            cbin = ((b[:, 0] + 1) // 2).astype(np.int32)
            row_list.append(np.where(rbin == 1, 0.7, 0.3))
            col_list.append(np.where(cbin == 1, 0.3, 0.7))

        # 先转换为numpy数组，再转换为张量（更高效）
        row_init = torch.tensor(np.array(row_list), dtype=torch.float32)  # [C,H]
        col_init = torch.tensor(np.array(col_list), dtype=torch.float32)  # [C,W]

        #row_init = torch.tensor(row_list, dtype=torch.float32)  # [C,H]
        #col_init = torch.tensor(col_list, dtype=torch.float32)  # [C,W]
        return row_init, col_init


class EnDecodingUNet(nn.Module):
    def __init__(self, H, W, C, select_mode=1, init_zigzag=None, random_seed=None):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.select_mode = select_mode

        if init_zigzag is None:
            init_zigzag = (select_mode in [1, 2, 3])

        self.pattern_generator = RowColPatternGenerator(
            H, W, C,
            select_mode=select_mode,
            init_zigzag=init_zigzag,
            random_seed=random_seed
        )
        self.Enchancing = EnchancingNet()


    def forward(self, x):
        B = x.size(0)
        patterns_bin = self.pattern_generator()  # [C,H,W]

        ''' ===== 1. 模拟 DMD 二值编码 ===== '''
        weight = patterns_bin.unsqueeze(1)  # [C,1,H,W]
        encoded = F.conv2d(x, weight)  # [B,C,H_out,W_out]
        C, H, W = encoded.shape[1:]

        ''' ===== 2. 编码结果全局标准化 ===== '''
        global_mean, global_std = encoded.mean(), encoded.std()
        encoded_norm = (encoded - global_mean) / global_std

        ''' ===== 3. 样本中心化处理 ===== '''
        sample_mean = encoded_norm.mean(dim=[1, 2, 3])  # [B]
        R = patterns_bin.sum(dim=[1, 2])  # [C]
        R_mean = R.mean()
        adj = ((sample_mean / R_mean) * 0.5).unsqueeze(1) * R.unsqueeze(0)  # [B,C]
        adj = adj.unsqueeze(2).unsqueeze(3)
        encoded_centered = encoded_norm - adj

        ''' ===== 4. pattern 去均值化处理 ===== '''
        pattern_mean = patterns_bin.mean(dim=0, keepdim=True)  # [1,H,W]
        patterns_zero_mean = patterns_bin - pattern_mean  # [C,H,W]

        ''' ===== 5. 核心 DGI 加权求和 ===== '''
        patterns_exp = patterns_zero_mean.unsqueeze(0).expand(B, -1, -1, -1)  # [B,C,H,W]
        weighted = patterns_exp * encoded_centered
        DGI_output = weighted.sum(dim=1, keepdim=True)  # [B,1,H,W]

        ''' ===== 6. DGI 可视化归一化 ===== '''
        DGI_min = DGI_output.amin(dim=[1, 2, 3], keepdim=True)
        DGI_max = DGI_output.amax(dim=[1, 2, 3], keepdim=True)
        DGI_norm = (DGI_output - DGI_min) / (DGI_max - DGI_min + 1e-8)

        ''' ===== 7. U-Net增强重建 ===== '''
        enhanced = self.Enchancing(DGI_norm)

        ''' ===== 8. 输出图像归一化 ===== '''
        enh_min = enhanced.amin(dim=[1, 2, 3], keepdim=True)
        enh_max = enhanced.amax(dim=[1, 2, 3], keepdim=True)
        enhanced_norm = (enhanced - enh_min) / (enh_max - enh_min + 1e-8)

        return patterns_bin, encoded, DGI_norm, DGI_output, enhanced_norm, enhanced

    def inference(self, encoded, return_all=False):
        """
        增强版推理方法，支持返回DGI和增强结果
        Args:
            encoded: 输入光强 [B,C,1,1]
            return_all: 是否返回所有中间结果（用于调试）
        Returns:
            默认返回 (dgi_result, enhanced_result)
            当return_all=True时返回元组 (patterns_bin, encoded, DGI_norm, enhanced_norm)
        """
        B = encoded.size(0)
        patterns_bin = self.pattern_generator()  # [C,H,W]

        # === 标准化处理 ===
        global_mean, global_std = encoded.mean(), encoded.std()
        encoded_norm = (encoded - global_mean) / (global_std + 1e-8)

        # === 样本中心化 ===
        sample_mean = encoded_norm.mean(dim=[1, 2, 3])  # [B]
        R = patterns_bin.sum(dim=[1, 2])  # [C]
        R_mean = R.mean()
        adj = ((sample_mean / R_mean) * 0.5).unsqueeze(1) * R.unsqueeze(0)  # [B,C]
        adj = adj.unsqueeze(2).unsqueeze(3)
        encoded_centered = encoded_norm - adj

        # === DGI核心计算 ===
        pattern_mean = patterns_bin.mean(dim=0, keepdim=True)  # [1,H,W]
        patterns_zero_mean = patterns_bin - pattern_mean  # [C,H,W]
        patterns_exp = patterns_zero_mean.unsqueeze(0).expand(B, -1, -1, -1)  # [B,C,H,W]

        weighted = patterns_exp * encoded_centered
        DGI_output = weighted.sum(dim=1, keepdim=True)  # [B,1,H,W]

        # === DGI归一化 ===
        DGI_min = DGI_output.amin(dim=[1, 2, 3], keepdim=True)
        DGI_max = DGI_output.amax(dim=[1, 2, 3], keepdim=True)
        DGI_norm = (DGI_output - DGI_min) / (DGI_max - DGI_min + 1e-8)

        # === U-Net增强 ===
        enhanced = self.Enchancing(DGI_norm)
        enh_min = enhanced.amin(dim=[1, 2, 3], keepdim=True)
        enh_max = enhanced.amax(dim=[1, 2, 3], keepdim=True)
        enhanced_norm = (enhanced - enh_min) / (enh_max - enh_min + 1e-8)

        # === 结果转换 ===
        def to_numpy(tensor):
            return tensor.squeeze(0).squeeze(0).cpu().numpy()

        if return_all:
            return (patterns_bin.cpu(),
                    encoded.squeeze().cpu(),
                    to_numpy(DGI_norm),
                    to_numpy(enhanced_norm))
        else:
            return to_numpy(DGI_norm), to_numpy(enhanced_norm)

''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''



''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' 辅助函数Unet-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''

class EnchancingNet(nn.Module):
    def __init__(self):
        super(EnchancingNet, self).__init__()

        self.Enchancing_Conv2d = UNet()                     # self.SRRes_net = SRResNet()

    def forward(self, x):
        Enchancing_outputs = self.Enchancing_Conv2d(x)

        return Enchancing_outputs

    ''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
    ''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
    ''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
    ''' UNet structure'''



class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()

            self.MaxPool_2d = nn.MaxPool2d(kernel_size=2)

            ''' ====================================================================================================='''
            self.UNet_Enhancing_1_DownLayer1 = nn.Sequential(
                # ''' ------------- DownLayer 1 ------------- '''
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1,
                                   output_padding=0),  # (B,1,128,128) -> (B,64,128,128)
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                # (B,64,128,128) -> (B,64,128,128)
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),

                # nn.MaxPool2d(kernel_size=2)                                                                # (B,64,128,128) -> (B,64,64,64)
            )

            self.UNet_Enhancing_2_DownLayer2 = nn.Sequential(
                # ''' ------------- DownLayer 2 ------------- '''
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                # (B,64,64,64) -> (B,128,64,64)
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                # (B,128,64,64) -> (B,128,64,64)
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),

                # nn.MaxPool2d(kernel_size=2)                                                                 # (B,128,64,64) -> (B,128,32,32)
            )

            self.UNet_Enhancing_3_DownLayer3 = nn.Sequential(
                # ''' ------------- DownLayer 3 ------------- '''
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                # (B,128,32,32) -> (B,256,32,32)
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                # (B,256,32,32) -> (B,256,32,32)
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),

                # nn.MaxPool2d(kernel_size=2)                                                                 # (B,256,32,32) -> (B,256,16,16)
            )

            self.UNet_Enhancing_4_DownLayer4 = nn.Sequential(
                # ''' ------------- DownLayer 4 ------------- '''
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                # (B,256,16,16) -> (B,512,16,16)
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                # (B,512,16,16) -> (B,512,16,16)
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),

                # nn.MaxPool2d(kernel_size=2)                                                                 # (B,512,16,16) -> (B,512,8,8)
            )

            self.UNet_Enhancing_5_BottomLayer = nn.Sequential(
                # ''' ------------- Bottomlayer  ------------- '''
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                # (B,512,8,8) -> (B,1024,8,8)
                nn.BatchNorm2d(num_features=512),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                # (B,1024,8,8) -> (B,1024,8,8)
                nn.BatchNorm2d(num_features=512),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0,
                                   output_padding=0),  # (B,1024,8,8) -> (B,512,16,16)
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),
            )

            self.UNet_Enhancing_6_UpLayer1 = nn.Sequential(
                # ''' --------------- UpLayer 1 --------------- '''
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
                # (B,1024,16,16) -> (B,512,16,16)
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                # (B,512,16,16) -> (B,512,16,16)
                nn.BatchNorm2d(num_features=256),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0,
                                   output_padding=0),  # (B,512,16,16) -> (B,256,32,32)
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),
            )

            self.UNet_Enhancing_7_UpLayer2 = nn.Sequential(
                # ''' --------------- UpLayer 2 --------------- '''
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                # (B,512,32,32) -> (B,256,32,328)
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                # (B,256,32,32) -> (B,256,32,32)
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0,
                                   output_padding=0),  # (B,256,32,32) -> (B,128,64,64)
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),
            )

            self.UNet_Enhancing_8_UpLayer3 = nn.Sequential(
                # ''' --------------- UpLayer 3 --------------- '''
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                # (B,256,64,64) -> (B,128,64,64)
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                # (B,128,64,64) -> (B,128,64,64)
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0,
                                   output_padding=0),  # (B,128,64,64) -> (B,64,128,128)
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),
            )

            self.UNet_Enhancing_9_UpLayer4 = nn.Sequential(
                # ''' --------------- UpLayer 4 --------------- '''
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                # (B,128,128,128) -> (B,64,128,128)
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                # (B,64,128,128) -> (B,64,128,128)
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                # (B,64,128,128) -> (B,1,128,128)
                nn.BatchNorm2d(num_features=1),
                nn.LeakyReLU()
            )

        ''' ====================================================================================================='''

        def forward(self, x):
            down_outputs = []
            # ---------------------------------------
            UNet_Enhancing_1_DownLayer1_outputs = self.UNet_Enhancing_1_DownLayer1(x)
            down_outputs.append(UNet_Enhancing_1_DownLayer1_outputs)
            UNet_Enhancing_1_DownLayer1_outputs = self.MaxPool_2d(UNet_Enhancing_1_DownLayer1_outputs)

            # ---------------------------------------
            UNet_Enhancing_2_DownLayer2_outputs = self.UNet_Enhancing_2_DownLayer2(UNet_Enhancing_1_DownLayer1_outputs)
            down_outputs.append(UNet_Enhancing_2_DownLayer2_outputs)
            UNet_Enhancing_2_DownLayer2_outputs = self.MaxPool_2d(UNet_Enhancing_2_DownLayer2_outputs)

            # ---------------------------------------
            UNet_Enhancing_3_DownLayer3_outputs = self.UNet_Enhancing_3_DownLayer3(UNet_Enhancing_2_DownLayer2_outputs)
            down_outputs.append(UNet_Enhancing_3_DownLayer3_outputs)
            UNet_Enhancing_3_DownLayer3_outputs = self.MaxPool_2d(UNet_Enhancing_3_DownLayer3_outputs)

            # ---------------------------------------
            UNet_Enhancing_4_DownLayer4_outputs = self.UNet_Enhancing_4_DownLayer4(UNet_Enhancing_3_DownLayer3_outputs)
            down_outputs.append(UNet_Enhancing_4_DownLayer4_outputs)
            UNet_Enhancing_4_DownLayer4_outputs = self.MaxPool_2d(UNet_Enhancing_4_DownLayer4_outputs)

            # =======================================
            UNet_Enhancing_5_BottomLayer_outputs = self.UNet_Enhancing_5_BottomLayer(
                UNet_Enhancing_4_DownLayer4_outputs)

            # =======================================
            skip = down_outputs[-1]
            UNet_Enhancing_5_BottomLayer_outputs = torch.cat([UNet_Enhancing_5_BottomLayer_outputs, skip], dim=1)
            UNet_Enhancing_6_UpLayer1_outputs = self.UNet_Enhancing_6_UpLayer1(UNet_Enhancing_5_BottomLayer_outputs)

            # ---------------------------------------
            skip = down_outputs[-2]
            UNet_Enhancing_6_UpLayer1_outputs = torch.cat([UNet_Enhancing_6_UpLayer1_outputs, skip], dim=1)
            UNet_Enhancing_7_UpLayer2_outputs = self.UNet_Enhancing_7_UpLayer2(UNet_Enhancing_6_UpLayer1_outputs)

            # ---------------------------------------
            skip = down_outputs[-3]
            UNet_Enhancing_7_UpLayer2_outputs = torch.cat([UNet_Enhancing_7_UpLayer2_outputs, skip], dim=1)
            UNet_Enhancing_8_UpLayer3_outputs = self.UNet_Enhancing_8_UpLayer3(UNet_Enhancing_7_UpLayer2_outputs)

            # ---------------------------------------
            skip = down_outputs[-4]
            UNet_Enhancing_8_UpLayer3_outputs = torch.cat([UNet_Enhancing_8_UpLayer3_outputs, skip], dim=1)
            UNet_Enhancing_9_UpLayer4_outputs = self.UNet_Enhancing_9_UpLayer4(UNet_Enhancing_8_UpLayer3_outputs)

            # UNet_Enhancing_9_UpLayer4_outputs_Normalization = (UNet_Enhancing_9_UpLayer4_outputs - torch.amin(UNet_Enhancing_9_UpLayer4_outputs, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3)) / (torch.amax(UNet_Enhancing_9_UpLayer4_outputs, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3) - torch.amin(UNet_Enhancing_9_UpLayer4_outputs, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3))

            ''' ====================================================================================================='''
            return UNet_Enhancing_9_UpLayer4_outputs
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''



# def main():
#     EncodingDecodingNet_ = EnDecodingUNet()
#     EncodingDecodingNet_input = torch.ones((64, 1, 32, 32))
#     Patterns, Encoding_Outputs, decoding_outputs, Decoding_outputs, enchancing_outputs, Enchancing_outputs = EncodingDecodingNet_(EncodingDecodingNet_input)
#     print(Enchancing_outputs.shape)
#     print(Enchancing_outputs.type)

def main():
    import matplotlib.pyplot as plt

    # 模拟输入维度
    H, W, C = 32, 32, 1024

    # 创建模型（使用 Hadamard Zigzag 初始化）
    model = EnDecodingUNet(H, W, C, init_zigzag=True)
    model.eval()

    # 获取 pattern
    with torch.no_grad():
        patterns = model.pattern_generator()  # [C, H, W]

    # 取前12个 pattern 可视化并标数字
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for idx in range(12):
        ax = axes[idx // 4, idx % 4]
        pattern = patterns[idx].cpu().numpy()

        # 二值化显示图像（颜色）
        ax.imshow(pattern, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Pattern {idx+1}')
        ax.axis('off')

        # 在每个位置标上0或1（取值大于0.5为1，否则为0）
        for i in range(H):
            for j in range(W):
                val = 1 if pattern[i, j] > 0.5 else 0
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=6, color='red' if val == 0 else 'blue')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

