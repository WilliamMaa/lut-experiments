# v8 并发实验：我（Agent）的问题清单

写于 v2026-10-04t repro 再次崩溃之后。不辩解，只列问题。

## 1. 核心错误：三个地方的"块单位"各自硬编码，从不互相校验

同一份代码里，"一个块是多少 token"以三种方式得到，且没有任何一处验证它们一致：

- `impl.py:73`：`bs = kv_cache.shape[2]`（运行时从 pool 形状读，崩溃现场算出 32）
- `backend.py:123,155`：`bs = self.spec.block_size`（builder 从 spec 读）
- 调度器实际分配：崩溃步 dump 显示 8192 token 的新请求只拿到 8 个块（`block_ids=([1],[2],[3],[4..11])`），即调度器按另一套单位记账

后果：builder 切出 `blocks_len=256`（=8192/32，一个 chunk 的块数），而
impl 的 write-back 需要 `16384/32 = 512` 个块 → 越界。我在三处用了三种
假设，却从没写过一个 `assert spec.block_size == kv_cache.shape[2]`，
也没核对过调度器侧的块大小。换数据（32k prompt、混合架构 4 个 KV group）
就炸，这不是"运气不好"，是设计上就没有单位一致性这回事。

## 2. 每个版本都是在上一个启发式上打补丁，不是在推导

n → p → q → r → s → t，六连修，每一版都换一个新的启发式规则：

- n：`n_computed==0` 判新请求 → 并发下 async lag 每步误触发 2916 次
- p：改看 builder 块表 →
- q：按 seq_lens 切片 → 块数少一个 chunk
- r：改按 snap_len 切片 → 等长重发误继承
- s：加"严格变长才算同一请求" → 还是崩
- t：加越界 raise + 打日志 → 崩得更有仪式感了

没有一版是先写出"哪些量是守恒的、它们的数学关系是什么"再动代码。
全是"猜一个规则 → 实机跑 → 看崩不崩"。这就是用户说的 hardcode：
**不是某个魔法数字，是整个方法论的 guessing。**

## 3. 把"防护"当修复

t 版加的 OOB raise 只是把静默错写变成了显式崩溃。数字（L2=16384,
jb2_max=511, n_blk=256）在 raise 出来之前我就该从代码里算出来：
builder 的 `n_need = min(..., bt.shape[1], nblk)` 这个 min clamp 本身就
保证了切片可能不够 write-back 用——这不是需要实机才能发现的 bug，
是读一遍代码就能发现的。

## 4. min() clamp 掩盖矛盾

`backend.py:155`：`n_need = min(ceil((seen+C)/bs), bt.shape[1], nblk)`。
需要 768 个块，min 给 256，静默通过，后面才炸。凡是"需要 X 但只给 Y"
的地方，正确写法是 assert / raise 带全现场，不是 clamp。

## 5. 身份判定自造轮子且不可靠

用 `blocks[0]` 当 key、用"前缀匹配+变长规则"猜请求身份，本质是在
模拟调度器的 allocator 重发行为。vLLM 有 request_id，有正经的
per-request 生命周期回调。我选择了猜，然后在"等长重发""块复用"这些
allocator 边角行为上连踩三轮。

## 6. 设计与验证脱节

spec.py 注释写着"blocks 必须覆盖整个 prompt 到 V8_MAX_SEQ_TOKENS"，
但单请求 64k 扫描能过、并发 32k 就 OOB，说明这条不变量在并发路径上
根本没被建立过，而单请求"通过"让我以为它成立。单请求通过 ≠ 不变量
成立，我把一个 happy path 的绿当成了证明。

## 7. 让用户替我跑 DEBUG 循环

每修一版都要用户在 8 卡机上起一次服务（~80s）+ 跑 bench + 贴日志，
而我本可以从已有数字（scheduler dump 里的 block_ids 和
num_scheduled_tokens）直接推出块单位不一致。信息早就在用户贴的输出里，
是我之前没去算。

## 8. 工程纪律形同虚设

- 每次升版要手改 3 个文件（config.py / repro 脚本 / runbook），
  却没有任何一个自动化检查会在改错时红灯。
- 本地只有 `py_compile` + `bash -n`（语法检查），没有任何单测覆盖
  builder 的切片数学、身份规则、单位换算——而这正是全部 bug 所在。
- 改动说明（"v2026-10-04x: 干了什么"）越写越长，等于承认每次都在
  打补丁而不是修复。

## 9. 沟通方式

崩溃后先给长篇辩解和"让我再分析分析"，而不是先承认"数字对不上，
是单位假设错了"。runbook 里塞诊断故事，用户要求删了又删。
在已经被证明猜错的路线（n→t 六连）上继续迭代，而不是停下来重写。

## 下一步应该怎么做（不是又一版补丁）

1. 从调度器侧拿真值：块大小、每 group 块数记账方式，写成常量推导
   + 启动时 assert（spec.block_size == pool bs == 调度器单位），
   不一致直接拒绝启动，不许 clamp。
2. 块表切片宽度按 write-back 的真实需求（ceil((compact_len 上界)/bs)）
   推导，需求 > 可用时 raise，不 min。
3. 请求身份：放弃 blocks[0] 启发式，找 vLLM 0.19.1 里能拿到
   request_id 的正路（builder 输入或 attention metadata 的 req 序
   与 engine 侧 seq_id 的映射）。
4. 以上三条落实前，不再让用户跑任何 repro。
