#!/usr/bin/env python3
"""
select_100_prompts.py

从 candidate_prompts.jsonl 中按任务比例分层挑选 100 条 prompt。

用法：
  python -u select_100_prompts.py \
    --input candidate_prompts.jsonl \
    --output selected_100_prompts.jsonl \
    --n_select 100 \
    --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_select", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    prompts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    n_total = len(prompts)
    print(f"Loaded {n_total} candidate prompts")

    # 1. 统计任务分布
    task_counts = Counter(p.get("task", "unknown") for p in prompts)
    print("\nTask distribution in candidate pool:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")

    # 2. 按比例计算每个任务的配额
    task_quotas = {}
    for task, count in task_counts.items():
        task_quotas[task] = max(1, round(args.n_select * count / n_total))

    # 3. 微调配额，确保总和正好等于 n_select
    while sum(task_quotas.values()) > args.n_select:
        task = max(task_quotas, key=task_quotas.get)
        if task_quotas[task] > 1:
            task_quotas[task] -= 1
        else:
            break

    while sum(task_quotas.values()) < args.n_select:
        task = max(task_counts, key=task_counts.get)
        task_quotas[task] += 1

    print(f"\nTask quotas (total={sum(task_quotas.values())}):")
    for task, quota in sorted(task_quotas.items(), key=lambda x: -x[1]):
        print(f"  {task}: {quota}")

    # 4. 按任务分组
    task_to_indices = defaultdict(list)
    for i, p in enumerate(prompts):
        task_to_indices[p.get("task", "unknown")].append(i)

    # 5. 每个任务随机抽取配额数量
    selected = []
    selected_set = set()
    for task in sorted(task_quotas.keys(), key=lambda t: -task_counts[t]):
        indices = task_to_indices[task]
        random.shuffle(indices)
        quota = task_quotas[task]
        take = min(quota, len(indices))
        for i in indices[:take]:
            selected.append(prompts[i])
            selected_set.add(i)

    # 6. 如果配额不足（比如某个任务样本太少），从剩余中补充
    remaining = [i for i in range(n_total) if i not in selected_set]
    random.shuffle(remaining)
    while len(selected) < args.n_select and remaining:
        selected.append(prompts[remaining.pop()])

    # 7. 统计输出
    print(f"\nSelected {len(selected)} prompts")

    selected_task_counts = Counter(p.get("task", "unknown") for p in selected)
    print("\nTask counts in selected:")
    for task, count in sorted(selected_task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count} (pool: {task_counts[task]})")

    lang_counts = Counter(p.get("language", "unknown") for p in selected)
    print("\nLanguage counts in selected (top 20):")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {lang}: {count}")

    print("\nTarget length distribution in selected:")
    length_bins = [0, 512, 1024, 2048, 4096, 100000]
    for lo, hi in zip(length_bins[:-1], length_bins[1:]):
        cnt = sum(1 for p in selected if lo <= p.get("target_length", 2048) < hi)
        print(f"  [{lo}, {hi}): {cnt}")

    # 8. 保存
    with open(args.output, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
