"""
生成模拟数据 CSV，每列服从不同的概率分布，用于测试分布拟合分析能力。
依赖: numpy
"""

import numpy as np

N = 10_000
np.random.seed(42)

data = {
    # ===== LinearGAM(Gaussian) 分支 =====
    # 条件: 不满足 Poisson (非整数或有负或 mean>=var) 且不满足 Gamma (有非正或 var<=mean)

    # 正态含负值 → 必然 Gaussian
    "product_quality_score": np.random.normal(loc=45, scale=15, size=N),

    # 正态围绕0 → 约一半负值
    "temperature_deviation_c": np.random.normal(loc=0, scale=3, size=N),

    # Laplace 对称厚尾 → 有负值
    "hedge_fund_return_pct": np.random.laplace(loc=0.2, scale=1.5, size=N),

    # t(df=4) 厚尾 → 有负值
    "stock_return_pct": np.random.standard_t(df=4, size=N) * 2 + 0.1,

    # 二项计数但 mean > var（欠离散）→ 不满足 Poisson(mean<var) 也不满足 Gamma(非全正)
    "survey_satisfied_count": np.random.binomial(n=100, p=0.3, size=N).astype(float),

    # ===== GammaGAM 分支 =====
    # 条件: 全正 + var > mean

    # Gamma (shape=5, scale=10)
    "daily_rainfall_mm": np.random.gamma(shape=5, scale=10, size=N),

    # Gamma (shape=2, scale=50) —— 重尾
    "insurance_claim_amount": np.random.gamma(shape=2, scale=50, size=N),

    # 指数 (scale=30) —— Gamma 特例
    "customer_wait_seconds": np.random.exponential(scale=30, size=N),

    # 卡方 (df=6) —— Gamma(k=3, θ=2)
    "server_response_time_ms": np.random.chisquare(df=6, size=N) * 5,

    # Weibull (shape=2.5)
    "component_lifetime_hours": np.random.weibull(a=2.5, size=N) * 50,

    # Pareto (shape=3) —— 重尾
    "household_income_k": np.random.pareto(a=3, size=N) * 10 + 1,

    # 对数正态 (μ=0, σ=0.5) —— 右偏
    "asset_price": np.random.lognormal(mean=0, sigma=0.5, size=N) * 100,

    # Beta (α=2, β=5)*200 —— 有界左偏
    "defect_rate_pct": np.random.beta(a=2, b=5, size=N) * 200,

    # 均匀 [0,200] —— 全正且 var >> mean，命中 Gamma
    "sensor_reading_mv": np.random.uniform(low=0, high=200, size=N),

    # ===== PoissonGAM 分支 =====
    # 条件: 非负整数 + mean < var（过离散）

    # 负二项 (n=5, p=0.3) —— 过离散计数
    "maintenance_ticket_count": np.random.negative_binomial(n=5, p=0.3, size=N).astype(float),

    # 负二项 (n=2, p=0.15) —— 强过离散
    "product_defect_count": np.random.negative_binomial(n=2, p=0.15, size=N).astype(float),

    # 泊松 (λ=15) —— mean≈var，边界情况
    "hourly_visitor_count": np.random.poisson(lam=15, size=N).astype(float),

    # 零膨胀泊松 —— 30%概率为0，过离散
    "accident_report_count": np.where(
        np.random.random(N) < 0.3, 0,
        np.random.poisson(lam=8, size=N)
    ).astype(float),
}

# 保留几位小数
for k, v in data.items():
    data[k] = np.round(v, 4)

# 写入 CSV
csv_path = "d:/for_fun_project/glacier/project/dist_test_data.csv"
header = ",".join(data.keys())
rows = np.column_stack(list(data.values()))
np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.4f")

print(f"已生成 {csv_path}，{N} 行 x {len(data)} 列")
print(f"列名及分布: {list(data.keys())}")