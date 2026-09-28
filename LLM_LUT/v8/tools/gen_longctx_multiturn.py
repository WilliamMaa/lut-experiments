#!/usr/bin/env python3
"""Generate synthetic long-context multi-turn workloads for v8 concurrency
serving benchmarks (docs/20 step 2, docs/21).

Documents are assembled from two block types:

  record blocks  — business-review paragraphs carrying traceable facts:
                   person + title + region + multi-digit figures (amounts,
                   percentages, counts) + dates. Tokenizers split multi-digit
                   numbers into single-digit tokens, so these facts directly
                   exercise M4 span-aware selection.
  filler blocks  — industry/process narrative with NO arabic numerals and no
                   record names, so filler never collides with ground-truth
                   substring checks and acts as compressible noise.

Records are spread evenly across the document (front / middle / back) so
questions at every turn hit facts at every depth. 8 questions per document,
mixed types:

  factoid            answer is a name/role/region span in the document
  digit_span         answer is a multi-digit figure (M4-sensitive)
  multi_instruction  question carries 2-3 constraints; ALL listed ground
                     truths must appear in the answer (instruction following)

Ground truths are verified to occur verbatim in the document at generation
time; the harness's fact_accuracy only checks model OUTPUT, so a gt list in
the JSONL means AND (all strings must appear).

Length calibration: with --tokenizer_path the document is tokenized and
filler is added/removed until the token count is within --tol of
--target_tokens. Without a tokenizer, --chars-per-token (default 1.6 for
Chinese Qwen BPE) gives a rough char target; run with the tokenizer on the
remote machine for exact calibration.

Output: one JSONL line per document:
  {"document": str, "questions": [str], "answers": [str | list[str]],
   "qtype": ["factoid"|"digit_span"|"multi_instruction"]}
"""

import argparse
import json
import random
from pathlib import Path

SURNAMES = "张王李赵刘陈杨黄周吴徐孙马朱胡郭何林罗郑梁谢宋唐许韩冯邓曹彭"
GIVEN = ["明", "华", "强", "丽", "洋", "斌", "婷", "伟", "芳", "娜", "磊", "静",
         "军", "磊", "燕", "鹏", "飞", "敏", "洁", "勇", "艳", "杰", "涛", "超"]
REGIONS = ["亚太区", "北美区", "欧洲区", "拉美区", "中东非区", "大中华区"]
DEPT_TOPICS = ["渠道", "产品", "技术", "运营", "市场", "客户成功", "供应链", "财务"]

METRICS = [
    ("收入", "亿元", (8, 99), 1),
    ("毛利率", "%", (21.0, 48.0), 0),
    ("运营利润率", "%", (6.0, 24.0), 0),
    ("客户续约率", "%", (80.0, 99.0), 0),
    ("同比增速", "%", (-5.0, 35.0), 0),
    ("新增合作伙伴", "家", (12, 120), 1),
    ("覆盖市场", "个", (3, 40), 1),
    ("季度活跃用户", "万", (50, 900), 1),
    ("研发预算", "千万元", (2, 60), 1),
    ("库存周转天数", "天", (15, 90), 1),
]

FILLER_TEMPLATES = [
    "会议还就{topic}条线的年度复盘进行了讨论，与会同事一致认为，当前阶段的重点是把已有流程做扎实，"
    "减少跨部门协作中的信息损耗，把经验文档化，让后续团队能够快速接手。",
    "{topic}板块随后汇报了上一阶段的推进情况，整体节奏符合预期，但在个别环节仍存在响应偏慢的问题，"
    "与会者建议把相关评审前置，并在下个周期开始前完成责任分工。",
    "围绕{topic}方向，主持人提示大家关注长期投入产出比，避免为了短期指标牺牲基础能力建设，"
    "同时要求各组把风险台账更新到最新版本并同步给相关方。",
    "在自由讨论环节，多位同事就{topic}的协作流程提出了改进建议，包括固定同步频次、明确升级路径、"
    "以及对历史决策保留完整的背景记录，便于新成员理解来龙去脉。",
    "关于{topic}的后续安排，会议决定维持现有节奏，先完成手头承诺事项，再在季度末统一回顾，"
    "期间如有阻塞问题，按既定渠道上报，不在会上展开个案讨论。",
    "{topic}组补充说明了近期的人员安排，强调目前梯队基本稳定，短期不会有大的调整，"
    "建议其他部门在排期时把这一约束纳入考虑，避免重复承诺。",
    "会议最后重申了文档与数据口径的统一要求：所有对外材料以最近一次评审通过的版本为准，"
    "历史版本仅作存档，不再作为决策依据，各组需要在下周之内完成自查。",
    "主持人总结时指出，本季度整体执行质量较上季有所改善，但细节打磨仍有空间，"
    "希望大家在保持进度的同时，把评审意见落到实处，而不是停留在纪要层面。",
]

INTRO_TEMPLATES = [
    "{year}年{quarter}季度经营回顾会议纪要",
    "{year}年{quarter}季度业务评审会纪要",
    "{year}年度{quarter}季度管理例会纪要",
]

SECTION_HEADERS = [
    "一、分区业务回顾", "二、重点指标通报", "三、专项讨论", "四、后续行动项",
    "五、风险提示", "六、资源协调", "七、流程改进", "八、自由讨论",
]


def _fmt_num(value, is_int):
    return str(int(round(value))) if is_int else f"{value:.1f}"


def make_record(rng, year, quarter, region):
    """One record block + its fact registry. Region is assigned by the caller
    (unique per document) so region-keyed questions are unambiguous."""
    name = rng.choice(SURNAMES) + rng.choice(GIVEN) + (rng.choice(GIVEN) if rng.random() < 0.3 else "")
    topic = rng.choice(DEPT_TOPICS)
    m1_name, m1_unit, m1_range, m1_int = rng.choice(METRICS)
    m1_val = rng.uniform(*m1_range) if not m1_int else rng.randint(*m1_range)
    m1_str = _fmt_num(m1_val, m1_int) + m1_unit
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date_str = f"{year}年{month}月{day}日"

    facts = []
    # Second metric with a different name, multi-digit friendly.
    m2_name, m2_unit, m2_range, m2_int = rng.choice(
        [m for m in METRICS if m[0] != m1_name])
    m2_val = rng.uniform(*m2_range) if not m2_int else rng.randint(*m2_range)
    m2_str = _fmt_num(m2_val, m2_int) + m2_unit

    block = (
        f"{region}负责人{name}在{topic}专项 review 中确认，{quarter}该分区{m1_name}为{m1_str}，"
        f"相关数据已经财务复核。她同时补充了{m2_name}的最新读数{m2_str}，"
        f"并指出该口径自{date_str}起生效，后续各期保持同口径可比。"
    )
    facts.append({"kind": "person", "q": f"{quarter}{region}的负责人是谁？", "gt": [name]})
    facts.append({"kind": "digit", "q": f"{quarter}{region}的{m1_name}是多少？",
                  "gt": [m1_str]})
    facts.append({"kind": "digit", "q": f"{region}最新披露的{m2_name}是多少？",
                  "gt": [m2_str]})
    facts.append({"kind": "multi",
                  "q": f"请同时给出{region}的{m1_name}和{m2_name}。",
                  "gt": [m1_str, m2_str]})
    facts.append({"kind": "factoid",
                  "q": f"{region}的{m2_name}口径从哪一天起生效？",
                  "gt": [date_str]})
    return block, facts


def make_filler(rng):
    k = rng.randint(2, 3)
    parts = [rng.choice(FILLER_TEMPLATES).format(topic=rng.choice(DEPT_TOPICS))
             for _ in range(k)]
    return "".join(parts)


def build_document(rng, target_tokens, tokenizer, tol, chars_per_token, year, quarter):
    """Assemble one document; calibrate length with filler blocks.

    Length is driven by a char estimate during assembly (O(1) per append via
    a running counter), then verified with the real tokenizer if one was
    provided; a mismatch is nudged by adding/removing filler blocks.
    """
    title = rng.choice(INTRO_TEMPLATES).format(year=year, quarter=quarter)
    # Regions unique per document: region-keyed questions stay unambiguous.
    # 6 regions x 5 facts each = 30 facts, comfortably covering 8 questions.
    n_records = len(REGIONS)
    regions = rng.sample(REGIONS, n_records)
    registry = [make_record(rng, year, quarter, regions[i]) for i in range(n_records)]
    order = list(range(n_records))
    rng.shuffle(order)
    facts = [f for _, fs in registry for f in fs]

    char_target = target_tokens * chars_per_token
    blocks = [title, ""]
    n_chars = len(title) + 1

    def add(block):
        nonlocal n_chars
        blocks.append(block)
        n_chars += len(block) + 1

    for rec_idx in order:
        add(registry[rec_idx][0])
        add("")
    # Filler only AFTER all records are placed, otherwise early records
    # trigger the fill loop and the remaining records blow past the target.
    while n_chars < char_target:
        add(make_filler(rng))
        add("")
    doc = "\n".join(blocks)

    # Exact token calibration when a tokenizer is available. Overshoot from
    # the last filler append is a few hundred tokens (<1% at 32k); nudge with
    # single filler add/remove until within tolerance.
    if tokenizer is not None:
        def ntokens(text):
            return len(tokenizer(text, add_special_tokens=False).input_ids)

        for _ in range(20):
            t = ntokens(doc)
            if target_tokens * (1 - tol) <= t <= target_tokens * (1 + tol):
                break
            if t < target_tokens * (1 - tol):
                extra = make_filler(rng)
                doc = doc + "\n\n" + extra
            else:
                # Drop the last filler paragraph (keeps all records intact).
                head, sep, _ = doc.rpartition("\n\n")
                if not sep:
                    break
                doc = head
        final_tokens = ntokens(doc)
    else:
        final_tokens = len(doc) / chars_per_token
    return doc, facts, final_tokens


def compose_questions(rng, facts):
    """Pick 8 questions: >=2 digit_span, >=2 multi_instruction, rest factoid."""
    digits = [f for f in facts if f["kind"] == "digit"]
    multis = [f for f in facts if f["kind"] == "multi"]
    persons = [f for f in facts if f["kind"] == "person"]
    others = [f for f in facts if f["kind"] == "factoid"]
    rng.shuffle(digits), rng.shuffle(multis), rng.shuffle(persons), rng.shuffle(others)

    chosen = []
    chosen += digits[:3]
    chosen += multis[:2]
    chosen += persons[:2]
    for f in others:
        if len(chosen) >= 8:
            break
        chosen.append(f)
    # Backfill from any pool if short.
    pool = digits + multis + persons + others
    for f in pool:
        if len(chosen) >= 8:
            break
        if f not in chosen:
            chosen.append(f)
    rng.shuffle(chosen)

    qtype_map = {"digit": "digit_span", "multi": "multi_instruction",
                 "person": "factoid", "factoid": "factoid"}
    return ([f["q"] for f in chosen],
            [f["gt"] for f in chosen],
            [qtype_map[f["kind"]] for f in chosen])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-tokens", type=int, default=32768)
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--questions-per-doc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--tokenizer-path", default=None,
                        help="Qwen tokenizer for exact token calibration (recommended)")
    parser.add_argument("--chars-per-token", type=float, default=1.6,
                        help="fallback Chinese chars/token ratio when no tokenizer")
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    year = 2025
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for d in range(args.num_docs):
            quarter = quarters[d % 4]
            doc, facts, ntokens = build_document(
                rng, args.target_tokens, tokenizer, args.tol,
                args.chars_per_token, year, quarter,
            )
            questions, answers, qtypes = compose_questions(rng, facts)
            questions = questions[:args.questions_per_doc]
            answers = answers[:args.questions_per_doc]
            qtypes = qtypes[:args.questions_per_doc]

            # Ground truths must be grounded in the document.
            for gt_list in answers:
                for gt in gt_list:
                    assert gt in doc, f"gt {gt!r} not in document {d}"
            digit_frac = sum(1 for q in qtypes if q == "digit_span") / len(qtypes)
            obj = {
                "document": doc,
                "questions": questions,
                "answers": answers,
                "qtype": qtypes,
                "meta": {"target_tokens": args.target_tokens,
                         "actual_tokens": round(ntokens, 1),
                         "doc_index": d},
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1
            print(f"[gen] doc {d}: tokens={ntokens:.0f} (target {args.target_tokens}), "
                  f"questions={len(questions)}, digit_span={digit_frac:.0%}")

    print(f"[gen] wrote {n_written} docs -> {out_path}")


if __name__ == "__main__":
    main()
