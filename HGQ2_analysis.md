Now I have all the information needed. Here is the detailed summary.

---

## HGQ2 仓库深度分析报告

---

### 1. 整体项目结构

```
HGQ2/
├── src/
│   ├── hgq/                          # 核心库
│   │   ├── __init__.py               # 包入口
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── layer.py              # LayerConfigScope (beta0, enable_ebops 等全局配置)
│   │   │   └── quantizer.py          # 重新导出 quantizer/config.py
│   │   ├── quantizer/
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # QuantizerConfig, QuantizerConfigScope (量化配置核心)
│   │   │   ├── quantizer.py          # Quantizer Layer (包装内部量化器的 Keras 层)
│   │   │   └── internal/
│   │   │       ├── __init__.py
│   │   │       ├── base.py           # TrainableQuantizerBase, DefaultBitwidthMapper, round_conv (STE)
│   │   │       ├── fixed_point_quantizer.py  # KBI / KIF 定点数量化器
│   │   │       └── float_point_quantizer.py  # MiniFloat 量化器 (alpha)
│   │   ├── layers/
│   │   │   ├── __init__.py           # 导出所有量化层
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py           # QLayerMeta (元类), QLayerBase, QLayerBaseSingleInput
│   │   │   │   ├── dense.py          # QDense, QBatchNormDense
│   │   │   │   └── einsum_dense.py   # QEinsumDense
│   │   │   ├── conv.py               # QConv1D, QConv2D, QConv3D
│   │   │   ├── activation.py         # QUnaryFunctionLUT, QAffinedUnaryFunctionLUT
│   │   │   ├── softmax.py            # QSoftmax (位精确 softmax)
│   │   │   ├── batch_normalization.py # QBatchNormalization
│   │   │   ├── pooling.py            # QMaxPool, QAvgPool, QGlobal*Pool
│   │   │   ├── einsum_dense_batchnorm.py # QEinsumDenseBatchnorm
│   │   │   ├── table/                # 基于查表的层 (LUT-based)
│   │   │   │   ├── dense.py          # QDenseT
│   │   │   │   └── conv.py           # QConvT1D, QConvT2D
│   │   │   ├── attn/
│   │   │   │   ├── mha.py            # QMultiHeadAttention
│   │   │   │   ├── linformer.py
│   │   │   │   └── salt.py
│   │   │   ├── ops/                  # QAdd, QMultiply, QEinsum, QDot, QSubtract...
│   │   │   └── rnn/                  # QSimpleRNN, QGRU
│   │   ├── constraints/
│   │   │   └── __init__.py           # MinMax, Min, Max, Constant
│   │   ├── regularizers/
│   │   │   └── __init__.py           # MonoL1
│   │   ├── utils/
│   │   │   ├── minmax_trace.py       # trace_minmax (校准工具)
│   │   │   ├── misc.py               # gather_vars_to_kwargs
│   │   │   ├── dataset.py
│   │   │   └── sugar/                # 便利工具
│   │   │       ├── ebops.py          # FreeEBOPs callback
│   │   │       ├── beta_scheduler.py # BetaScheduler, PieceWiseSchedule
│   │   │       ├── beta_pid.py       # BetaPID (PID 控制 beta)
│   │   │       └── ...
│   │   ├── _alkaid_keras_plugin/     # hls4ml/da4ml 插件
│   │   └── _dais_tracer/             # Dais tracer 插件
│   └── qkeras/                       # QKeras 兼容层 (alpha)
├── tests/                            # 测试套件
├── example/                          # 示例 (git-ignored, notebook)
├── docs/                             # Sphinx 文档
├── pyproject.toml
└── README.md
```

---

### 2. HGQ 是什么，解决什么问题

**HGQ = High Granularity Quantization (高粒度量化)**

HGQ2 是一个基于 Keras v3 的**量化感知训练 (QAT)** 框架，目标是将神经网络部署到 **FPGA 边缘设备**上用于实时推理。支持 TensorFlow/JAX/PyTorch 多后端。

**核心思想**: 通过梯度自动优化每个权重和每个激活值的最佳位宽 (bitwidth)，而不是手工调参。

关键特性:
- **高粒度**: 支持 per-weight、per-activation 的位宽优化，可细到每个权重值和每个激活值的不同位宽
- **自动量化**: 位宽通过梯度自动学习，无需手动调参
- **WYSIWYG**: Keras 模型输出与 RTL (硬件) 模型输出一致 (受限于机器浮点精度)
- **EBOPs**: 精确的资源估算指标 (Effective Bit-Operations)，与 FPGA 实际资源消耗 (LUT + DSP) 高度相关

在 FAQ 中提到: HGQ 在某些应用中相比传统 AutoQkeras 方法可以实现 **高达 10x 的资源节省**。

与你们项目的**关键区别**:
- HGQ2 的 "LUT" 层 (`QDenseT`, `QConvT`) 是用小型子网络 (EinsumDense) 生成对输入特征的**逐元素查表**——这更像是一个小型函数近似器，不是传统意义上的 LUT 查表。
- HGQ2 主要解决的是 **FPGA 量化部署** 问题，用 QAT 训练位宽；你们项目解决的是用 **LUT 查表替代神经网络计算** 来降低功耗。

---

### 3. 量化感知训练 (QAT) 的核心算法

HGQ2 的核心训练流程不是 "LAT" (LUT-Aware Training)，而是标准的 **量化感知训练 (QAT) + 可微位宽优化**。具体流程:

#### 3.1 全局配置 (LayerConfigScope)

```python
# 文件: src/hgq/config/layer.py
class GlobalConfig(TypedDict):
    beta0: float           # L1 正则化初始强度，控制资源消耗
    enable_ebops: bool     # 是否追踪 EBOPs
    enable_oq: bool        # 是否在层后插入输出量化器
    enable_iq: bool        # 是否在层前插入输入量化器

global_config = GlobalConfig(
    beta0=1e-5,
    enable_ebops=True,
    enable_oq=False,
    enable_iq=True,
)
```

#### 3.2 量化器配置 (QuantizerConfig)

```python
# 文件: src/hgq/quantizer/config.py
# 三种量化类型:

# 1. KBI: Keep(符号), Bits(总位数-不含符号), Integer(整数位数)
kbi_weight_default = KBIConfig(
    k0=True,           # 允许负值
    b0=8,              # 初始总位数
    i0=2,              # 初始整数位
    round_mode='RND',  # 舍入模式
    overflow_mode='SAT_SYM',  # 溢出模式 (对称饱和)
    bc=MinMax(0, 23),  # 位宽约束
    br=MonoL1(1e-8),   # 位宽 L1 正则化 (关键: 推动位宽降低)
    ...
)

# 2. KIF: Keep, Integer, Fraction
kif_datalane_default = KIFConfig(
    k0=True,
    i0=2,              # 整数位
    f0=6,              # 小数位
    overflow_mode='WRAP',  # 数据通常用 WRAP
    ic=MinMax(-23, 23),
    fc=MinMax(-24, 24),
    ir=MonoL1(1e-8),
    fr=MonoL1(1e-8),
    i_decay_speed=0.01,  # WRAP 模式下整数位的衰减速度
    ...
)

# 3. Float: MiniFloat (alpha quality)
```

**四种 "place" (量化器位置)**:
- `weight`: 权重量化，默认 `SAT_SYM`，`heterogeneous_axis=None` (全异构/per-weight)
- `datalane`: 数据/激活量化，默认 `WRAP`，`homogeneous_axis=(0,)` (batch 维度同构)
- `bias`: 偏置量化
- `table`: 查表层输出量化

#### 3.3 量化器 Scope 上下文管理器

```python
with (
    QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
    QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
    LayerConfigScope(enable_ebops=True, beta0=1e-5),
):
    model = keras.Sequential([
        QConv2D(32, (3, 3), activation='relu'),
        QDense(10)
    ])
```

---

### 4. 前向传播：量化是如何施加的

#### 4.1 量化器的 call 方法

```python
# 文件: src/hgq/quantizer/quantizer.py
class Quantizer(Layer):
    def call(self, inputs, training=None):
        if self.scaler is not None:
            inputs = inputs / self.scaler           # 可选的前置缩放
        inputs = ops.cast(inputs, ops.dtype(inputs)) # 确保是 tensor
        outputs = self.quantizer.call(inputs, training=training)  # 核心量化
        if self.scaler is not None:
            outputs = outputs * self.scaler          # 恢复缩放
        if self.qnoise_factor is not None and training:
            # 可选的量化噪声混合: (1-noise)*fp + noise*quantized
            outputs = inputs + self.qnoise_factor * (outputs - inputs)
        if self.affine is not None:
            outputs = outputs * self.affine[0] + self.affine[1]
        return outputs
```

#### 4.2 定点数量化核心 (KBI / KIF)

```python
# 文件: src/hgq/quantizer/internal/fixed_point_quantizer.py
class FixedPointQuantizerBase(TrainableQuantizerBase):
    def call(self, inputs, training=None):
        k, i, f = self.kif   # 符号位、整数位、小数位
        k = self.bw_mapper.bw_to_x(k, ops.shape(inputs))  # 广播到输入形状
        i = self.bw_mapper.bw_to_x(i, ops.shape(inputs))
        f = self.bw_mapper.bw_to_x(f, ops.shape(inputs))
        # 调用无状态定点量化器 (来自 quantizers 库)
        ret = self.stateless_quantizer(inputs, k, i, f, training is True, self.seed_gen)
        if not training:
            # 推理模式下，零位宽将输出置零
            ret = ops.where(k + i + f > 0, ret, ops.zeros_like(ret))
        return ret
```

#### 4.3 WRAP 溢出模式的特殊处理

对于 `WRAP` 模式的激活量化器 (如 datalane)，有一个**自动整数位追踪**机制:

```python
# KIF WRAP 模式:
def call(self, inputs, training=None):
    if self.overflow_mode == 'WRAP' and self.trainable:
        f = self.bw_mapper.bw_to_x(self.f, ops.shape(inputs))
        rinputs = self.stateless_quantizer.round(inputs, f, stochastic, self.seed_gen)
        if training or training == 'tracing':
            _new_i = self.get_minimal_i(rinputs)   # 计算所需的最小整数位
            if training:
                # 训练时: 整数位逐步衰减 (防止突变)
                new_i = ops.stop_gradient(ops.maximum(self._i - self.i_decay_speed, _new_i))
                self._i.assign(new_i)
            else:
                # tracing 时: 取最大值
                new_i = ops.stop_gradient(ops.maximum(self.i, _new_i))
                new_k = self.get_any_k(rinputs)     # 更新符号位
                self._k.assign(new_k)
                self._i.assign(new_i)
            return rinputs
```

#### 4.4 QDense / QConv 如何集成量化

```python
# 文件: src/hgq/layers/core/dense.py
class QDense(QLayerBaseSingleInput, Dense):
    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)    # 输入量化
        x = ops.matmul(inputs, self.qkernel)               # 使用量化后的权重
        if self.bias is not None:
            x = ops.add(x, self.qbias)                     # 量化后的偏置
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def qkernel(self):
        return self.kq(self._kernel)   # 权重通过量化器

    @property
    def qbias(self):
        return self.bq(self.bias)      # 偏置通过量化器
```

---

### 5. 反向传播：STE (Straight-Through Estimator) 实现

#### 5.1 核心 STE: `round_conv`

```python
# 文件: src/hgq/quantizer/internal/base.py
@ops.custom_gradient
def round_conv(x):
    qx = ops.round(x)   # 前向: 标准 round

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return upstream    # 反向: 直接传递梯度 (STE!)

    return qx, grad
```

这就是标准的 STE: 前向传播时执行 `round(x)`，反向传播时梯度直接穿过 (identity gradient)。

`round_conv` 在以下关键位置被使用:
- `FixedPointQuantizerKBI.b`, `FixedPointQuantizerKBI.i`: 位宽参数的 round (位宽是连续训练但有约束的)
- `FixedPointQuantizerKIF.i`, `FixedPointQuantizerKIF.f`: 同样
- `FloatPointQuantizer.m`, `.e`, `.e0`: MiniFloat 参数
- 底层的 `stateless_quantizer` 中的 `round` 操作 (来自 `quantizers` 库)

#### 5.2 位宽正则化推动资源压缩

```python
# 文件: src/hgq/regularizers/__init__.py
class MonoL1(Regularizer):
    def __init__(self, l1: numbers):
        self.l1 = float(l1)
    def __call__(self, x):
        return self.l1 * ops.sum(x)  # L1 正则化: 推动位宽趋向零
```

这个 L1 正则化通过 `beta` 参数被加权:

```python
# 文件: src/hgq/layers/core/base.py (QLayerMeta.__call__ 中的 call wrapper)
def call(self, *args, **kwargs):
    r = original_call(self, *args, **kwargs)
    if (training or training == 'tracing') and self.enable_ebops:
        ebops = self._compute_ebops(*shapes) * self.ebops_factor
        self._ebops.assign(ops.cast(ebops, self._ebops.dtype))
        self.add_loss(ebops * self.beta)   # EBOPs 被加入到总 loss
    ...
```

**所以完整的训练目标 = 任务 loss + beta * EBOPs**。`beta` 通过 `BetaScheduler` 或 `BetaPID` 动态调整，实现资源-精度的 Pareto 最优搜索。

---

### 6. 所有量化相关的核心类

| 类名 | 文件 | 作用 |
|------|------|------|
| `Quantizer` | `quantizer/quantizer.py` | 通用量化器 Keras Layer，包装内部量化器 |
| `QuantizerConfig` | `quantizer/config.py` | 量化器配置 (KBI/KIF/Float/Dummy) |
| `QuantizerConfigScope` | `quantizer/config.py` | 上下文管理器，批量覆盖量化配置 |
| `TrainableQuantizerBase` | `quantizer/internal/base.py` | 所有内部量化器的抽象基类 |
| `FixedPointQuantizerKBI` | `quantizer/internal/fixed_point_quantizer.py` | KBI 定点量化: k + b + i -> k + i + (b-i) |
| `FixedPointQuantizerKIF` | `quantizer/internal/fixed_point_quantizer.py` | KIF 定点量化: k + i + f |
| `FloatPointQuantizer` | `quantizer/internal/float_point_quantizer.py` | MiniFloat 量化 (IEEE 754 类) |
| `DummyQuantizer` | `quantizer/internal/base.py` | 无操作量化器 (bypass) |
| `DefaultBitwidthMapper` | `quantizer/internal/base.py` | 位宽到张量的形状映射 |
| `round_conv` | `quantizer/internal/base.py` | STE 的 `ops.custom_gradient` 实现 |
| `QLayerBase` | `layers/core/base.py` | 所有量化层的基类，管理 EBOPs/beta/oq |
| `QLayerBaseSingleInput` | `layers/core/base.py` | 单输入量化层基类，管理 iq |
| `QLayerMeta` | `layers/core/base.py` | 元类：自动包装 call/build，注入 EBOPs 计算 |
| `MinMax`, `Min`, `Max`, `Constant` | `constraints/__init__.py` | 位宽参数约束 |
| `MonoL1` | `regularizers/__init__.py` | L1 正则化 (推动位宽压缩) |
| `LayerConfigScope` | `config/layer.py` | 层级别全局配置 (beta0, enable_ebops 等) |

---

### 7. 与现有模型的集成方式

HGQ2 采用 **直接替换** (drop-in replacement) 策略:

```python
# 标准 Keras:
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3)),
    keras.layers.Dense(10)
])

# HGQ2 替换:
from hgq.layers import QConv2D, QDense
model = keras.Sequential([
    QConv2D(32, (3,3), activation='relu'),   # 直接替换
    QDense(10)                                 # 直接替换
])
```

**底层机制 (QLayerMeta 元类)**:

```python
# 文件: src/hgq/layers/core/base.py
class QLayerMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # 1. 包装 call 方法: 自动计算 EBOPs + 应用输出量化器
        # 2. 包装 build 方法: 检查量化后权重是否 collapses
        original_call = cls.call
        @wraps(original_call)
        def call(self, *args, **kwargs):
            r = original_call(self, *args, **kwargs)
            if training and self.enable_ebops:
                ebops = self._compute_ebops(*shapes)
                self.add_loss(ebops * self.beta)   # 自动添加资源损失
            if not self.enable_oq:
                return r
            return self.oq(r, training=training)   # 自动添加输出量化
        cls.call = call
        return super().__call__(*args, **kwargs)
```

**支持的量化层完整列表** (来自 `layers/__init__.py`):
- 核心: `QDense`, `QBatchNormDense`, `QEinsumDense`
- 卷积: `QConv1D`, `QConv2D`, `QConv3D`
- 归一化: `QBatchNormalization`, `QEinsumDenseBatchnorm`
- 池化: `QMaxPool1D/2D/3D`, `QAvgPool1D/2D/3D`, `QGlobalAvgPool1D/2D/3D`, `QGlobalMaxPool1D/2D/3D`
- 激活/LUT: `QUnaryFunctionLUT`, `QAffinedUnaryFunctionLUT`
- Softmax: `QSoftmax` (位精确，含 exp/inv 子表)
- 注意力: `QMultiHeadAttention`, `QLinformerAttention`
- 运算: `QAdd`, `QMultiply`, `QSubtract`, `QDot`, `QEinsum`, `QSum`, `QMeanPow2`
- 查表: `QDenseT`, `QConvT1D`, `QConvT2D`
- RNN: `QSimpleRNN`, `QGRU`

**部署集成 (hls4ml/da4ml)**:

```python
from hgq.utils import trace_minmax
from hls4ml.converters import convert_from_keras_model

trace_minmax(model, x_test)   # 校准 WRAP 模式的整数位
hls_model = convert_from_keras_model(model, ...)
```

---

### 8. 配置系统与训练回调

#### 8.1 Beta 调度器

```python
# 文件: src/hgq/utils/sugar/beta_scheduler.py
class BetaScheduler(Callback):
    """按 epoch 调整 beta 值"""
    def __init__(self, beta_fn):
        ...

class PieceWiseSchedule:
    """分段调度: 支持 linear / log / constant 插值
    例: [(0, 0, 'linear'), (10, 1e-5, 'log'), (20, 1e-3, 'constant')]
    """
```

#### 8.2 PID 控制器 (自动达到目标 EBOPs)

```python
# 文件: src/hgq/utils/sugar/beta_pid.py
class BetaPID(BaseBetaPID):
    """用 PID 控制器自动调节 beta 以达到目标 EBOPs
    - target_ebops: 目标资源消耗
    - p/i/d: PID 增益
    - warmup: 前 N 个 epoch 不更新 beta
    - log: 是否在 log 空间控制 (推荐)
    - damp_beta_on_target: 达到目标后衰减 beta
    """
```

#### 8.3 trace_minmax (校准工具)

```python
# 文件: src/hgq/utils/minmax_trace.py
def trace_minmax(model, data, reset=True, batch_size=1024, verbose=False):
    """对 WRAP 模式的激活量化器，用校准数据集跟踪所需的最小整数位
    调用 model(data, training='tracing') 触发累加追踪
    """
```

#### 8.4 完整训练流程示例

```python
# 1. 配置量化
with QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM'):
    with QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='WRAP'):
        with LayerConfigScope(enable_ebops=True, beta0=1e-5):
            model = keras.Sequential([...])

# 2. 编译
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. 训练 (带 beta 调度)
ebops_cb = FreeEBOPs()
model.fit(x_train, y_train, epochs=15, callbacks=[ebops_cb])

# 4. 部署前校准
trace_minmax(model, x_test)
```

---

### 总结: HGQ2 与你们项目的对比

| 维度 | HGQ2 | 你们的项目 (LUT) |
|------|------|-------------------|
| **目标** | FPGA 量化部署，自动位宽优化 | CIM 设备上用 LUT 查表替代神经网络计算 |
| **核心方法** | QAT + 可微位宽学习 + STE | LUT 查表 (低维 query 直接索引参数残差) |
| **计算量** | 位宽优化减少计算量，但仍是矩阵乘法 | O(1) 查表，训练集大小无关 |
| **"LUT" 含义** | `QUnaryFunctionLUT` 是激活函数的量化查表; `QDenseT` 是小网络+查表 | 用于存储和检索神经网络权重/残差的查表 |
| **适用硬件** | FPGA (Vitis/Vivado) | CIM (存算一体) 设备 |

HGQ2 的 "LUT" (`QDenseT`, `QConvT`) 实际上是：用一个小型子网络 (EinsumDense with `tanh`) 对每个输入特征做函数近似，然后查表输出——这更像是一种**基于表的非线性映射**，而不是你们项目中的**权重参数 LUT 索引**。

HGQ2 中最值得你们参考的技术:
1. **STE 的 `ops.custom_gradient` 实现** (`round_conv`) -- 简洁高效
2. **Beta 调度 + PID 控制** 的资源-精度 Pareto 优化机制
3. **`QuantizerConfigScope` 上下文管理器** 的分层配置设计模式
4. **`QLayerMeta` 元类** 自动注入量化逻辑的方式 (不改动原始层代码)