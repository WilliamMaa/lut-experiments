
class FeatureLearnableLSH_SpatialNonLinearGatedGumbelIndexGenerator(nn.Module):
    """
    INDEX:Spatial non-linear Gated index generator with Gumbel-Softmax relaxation.
    Combines receptive field, dynamic gating, and smooth gradient flow.
    """
    def __init__(self, c_in: int, num_bits: int, base: int = 2):
        super().__init__()
        self.num_bits = num_bits
        self.base = base
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.SiLU(inplace=True)
        )
        
        hidden_dim = max(c_in, num_bits * 2)
        out_dim = num_bits if base == 2 else num_bits * base
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.hash_head = nn.Linear(hidden_dim, out_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, num_bits),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.spatial_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous() 
        
        hidden = self.mlp(x)
        logits = self.hash_head(hidden)
        scales = self.gate_head(hidden) # (B, H, W, num_bits)
        
        if self.base == 2:
            # 对于二分类，Gumbel-Softmax 可以通过对 logit 叠加 Gumbel 噪声并用 Sigmoid 逼近
            # 为了简便和稳定性，这里我们采用软化概率 + STE 结合温度
            probs = torch.sigmoid(logits / tau)
            if self.training:
                # 训练时使用 Gumbel 噪声
                u = torch.rand_like(logits)
                gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
                noisy_logits = (logits + gumbel_noise) / tau
                soft_bits = torch.sigmoid(noisy_logits)
            else:
                soft_bits = probs
                
            hard_bits = (soft_bits > 0.5).float()
            bits = hard_bits.detach() - soft_bits.detach() + soft_bits
            return bits, scales
        else:
            B, H, W, _ = logits.shape
            logits = logits.view(B, H, W, self.num_bits, self.base)
            
            if self.training:
                # 训练时使用 Gumbel-Softmax 替代 argmax + STE
                bits = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            else:
                # 推理时退化为普通的 argmax
                idx = torch.argmax(logits, dim=-1)
                bits = F.one_hot(idx, num_classes=self.base).float()
                
            return bits, scales



class FineGrainedMoELUT_Distiller(nn.Module):
    """
    LUT:
    符合「细粒度 MoE (Parameter-LUT)」底层逻辑的全新 LUT 蒸馏器：
    
    【核心逻辑转变】：
    原架构 (Feature LUT)：LUT 存储的是"特征结果"。前向时累加多张表的特征。
    新架构 (Parameter LUT)：LUT 存储的是"模型参数"。对于输入到输出层的一个 FFN 结构，
    不同的特征（像素）进来时，根据 Index 查出专属的 FFN 参数 (W1, W2)。
    
    【实现原理】：
    1. 为每个 bit 表构建存储 FFN 参数的 Embedding (包含 W1 和 W2)。
    2. 根据特征对应的 bits，查出所有表的参数并累加，组合出当前特征专属的 FFN 参数。
    3. 所有的特征依然通过一个"逻辑上的 FFN"结构进行前向计算，但每个特征使用的
       权重矩阵是不一样的，从而在保证特征处理结构不变的条件下，大幅提高拟合能力。
    """
    def __init__(self, num_bits: int = 16, c_in: int = 128, c_out: int = 128, base: int = 2):
        super().__init__()
        self.num_bits = num_bits
        self.base = base
        self.c_in = c_in
        self.c_out = c_out
        
        # 简化为单层 Linear 映射 (c_in -> c_out)
        # 每个表项存储的参数量即为 W 的大小: c_out * c_in
        self.w_size = self.c_out * self.c_in
        self.param_size = self.w_size
        
        # LUT 存储的不再是输出特征，而是 Linear 的网络参数
        self.idx_embs = nn.ModuleList([
            nn.Embedding(base, self.param_size) for _ in range(num_bits)
        ])
        
        # 初始化，使用较小的 std 以防止动态生成的参数值过大导致不收敛
        for emb in self.idx_embs:
            nn.init.normal_(emb.weight, std=0.02 / (num_bits ** 0.5))


    def forward(self, orig: torch.Tensor, bits: torch.Tensor, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = orig.shape
        
        # 1. 直接使用矩阵乘法完成所有查表，彻底消除循环和列表带来的显存开销
        if self.base == 2:
            # bits: (B*H*W, num_bits)
            bits_flat = bits.reshape(-1, self.num_bits)
            
            # 将 idx_embs 拼凑成一个大的权重矩阵: (num_bits, 2, param_size)
            all_weights = torch.stack([emb.weight for emb in self.idx_embs], dim=0)
            
            # w0: (num_bits, param_size), w1: (num_bits, param_size)
            w0 = all_weights[:, 0, :]
            w1 = all_weights[:, 1, :]
            
            # 计算基底和偏移量
            # sum_w0: (param_size,)
            sum_w0 = w0.sum(dim=0)
            # w_diff: (num_bits, param_size)
            w_diff = w1 - w0
            
            if scales is not None:
                # scales_flat: (B*H*W, num_bits)
                scales_flat = scales.reshape(-1, self.num_bits)
                # 显存优化：将三元 einsum 拆解为逐元素相乘 + 矩阵乘法
                # scaled_bits 只有 (B*H*W, num_bits)，占用不到 10MB
                scaled_bits = bits_flat * scales_flat 
                # (B*H*W, num_bits) @ (num_bits, param_size) -> (B*H*W, param_size)
                pred_params = sum_w0 + torch.matmul(scaled_bits, w_diff)
            else:
                # 一次性矩阵乘法: (B*H*W, num_bits) x (num_bits, param_size) -> (B*H*W, param_size)
                pred_params = sum_w0 + torch.matmul(bits_flat, w_diff)
                
        else:
            # bits: (B*H*W, num_bits, base)
            bits_flat = bits.reshape(-1, self.num_bits, self.base)
            
            # 将 idx_embs 拼凑成一个大的权重矩阵: (num_bits, base, param_size)
            all_weights = torch.stack([emb.weight for emb in self.idx_embs], dim=0)
            
            if scales is not None:
                scales_flat = scales.reshape(-1, self.num_bits, 1)
                bits_flat = bits_flat * scales_flat
                
            # 显存优化：展平多余维度直接使用 matmul，避免 einsum 寻找低效路径
            # bits_flat 展平: (B*H*W, num_bits * base)
            bits_flat_2d = bits_flat.reshape(-1, self.num_bits * self.base)
            # all_weights 展平: (num_bits * base, param_size)
            weights_2d = all_weights.reshape(self.num_bits * self.base, self.param_size)
            
            # (B*H*W, num_bits * base) @ (num_bits * base, param_size) -> (B*H*W, param_size)
            pred_params = torch.matmul(bits_flat_2d, weights_2d)
        
        # 2. 将聚合出的参数转换为单一权重矩阵 W
        # pred_params 的形状是 (B*H*W, param_size)
        
        # W 形状: (B*H*W, c_out, c_in)
        W_matrix = pred_params.view(-1, self.c_out, self.c_in)
        
        # 3. 准备输入特征 (B*H*W, c_in, 1)
        x_flat = orig.permute(0, 2, 3, 1).reshape(-1, self.c_in, 1)
        
        # 4. 执行单层 Linear 计算 (细粒度 MoE)
        # 每个像素使用自己专属的 W 进行 in-out 映射
        
        # OOM 的根源在 torch.bmm，当 B*H*W 很大时 (如 20万), 
        # (B*H*W, c_out, c_in) 与 (B*H*W, c_in, 1) 的 bmm 会占用极其恐怖的显存用于反向传播
        # 我们使用 torch.einsum 替代 bmm，并且利用 einsum 在特定情况下的显存优化
        out_flat = torch.einsum('boi,bi->bo', W_matrix, x_flat.squeeze(-1)).unsqueeze(-1)
        
        
        # 5. 还原回图像特征的形状 (B, c_out, H, W)
        out = out_flat.view(B, H, W, self.c_out).permute(0, 3, 1, 2).contiguous()

        return out