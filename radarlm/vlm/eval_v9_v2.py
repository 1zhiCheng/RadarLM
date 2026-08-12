"""v9 多模态雷达感知 QA 评价指标 (v3 - 严格版).

用户要求:
1. 拆 ask-class presence vs any-target presence (问"是否有行人"不能答"无目标")
2. 数值 > 5% 算错 (严格)
3. 疑似目标优先答目标, 漏检大惩罚
4. bbox 越界检查
5. 描述题数值检查
6. refusal/hallucination 独立
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


VIEW_PATTERNS = {
    'RD': [r'RD\s*视图', r'range[_\s-]?doppler', r'距离[\-—]?多普勒', r'\(RD\)'],
    'RA': [r'RA\s*视图', r'range[_\s-]?angle', r'距离[\-—]?角度', r'\(RA\)'],
    'AD': [r'AD\s*视图', r'angle[_\s-]?doppler', r'角度[\-—]?多普勒', r'\(AD\)'],
}

# 各视图尺寸 (H, W)
VIEW_SHAPES = {
    'RD': (256, 64),
    'RA': (256, 256),
    'AD': (256, 64),
}

CLASS_NAMES = ["汽车", "行人", "骑行者", "自行车"]
CLASS_TO_LETTER = {"汽车": "A", "行人": "B", "骑行者": "C", "自行车": "C"}


def detect_view(text: str) -> Optional[str]:
    text_lower = text.lower()
    positions = []
    for view, patterns in VIEW_PATTERNS.items():
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                positions.append((m.start(), view))
    if positions:
        positions.sort()
        return positions[0][1]
    return None


def detect_asked_class(question: str) -> Optional[str]:
    """从问题识别询问的类别 (汽车/行人/骑行者/自行车)."""
    q = question
    for cls in CLASS_NAMES:
        if cls in q:
            return cls
    # "是否有行人" 类
    if "行人" in q: return "行人"
    if "汽车" in q: return "汽车"
    if "骑行者" in q or "自行车" in q: return "骑行者"
    return None


def _has_class_in_text(text: str, cls: str) -> Optional[bool]:
    """检测 text 中是否说 "有 cls" / "无 cls" (考虑上下文).
    ★ v7.1: 支持单字母 GT/GEN ("A"=汽车, "B"=行人, "C"=骑行者, "D"=无目标).
    """
    text = text.strip()
    # v7.14: multi-choice 选项 (A.汽车\nB.行人\nC.骑行者\nD.无目标)
    # 遍历每个 option, 找 cls 子串
    import re
    options = re.split(r'\n[ABCD][\.\)、]', text)
    # options[0] = "A.汽车", options[1] = "行人" (B), options[2] = "骑行者" (C), options[3] = "无目标" (D)
    if len(options) >= 4:  # multi-choice 格式 (A/B/C/D 4个选项)
        for i, opt in enumerate(options[0:4]):
            opt_letter = chr(ord('A') + i)
            if opt_letter == 'D':
                continue
            # 匹配 cls 的各种变体 (e.g. "骑行者/自行车" matches "骑行者")
            if cls in opt or cls.split('/')[0] in opt:
                return True
        return False  # cls 不在 A/B/C → False

    # 单字母 GT/GEN: 整个 text 只是 "A"/"B"/"C"/"D" (可能带标点)
    letter_to_class = {"A": "汽车", "B": "行人", "C": "骑行者/自行车"}
    first_char = text[0] if text else ""
    if first_char in letter_to_class and len(text) <= 3:
        return letter_to_class[first_char] == cls
    if first_char == "D" and len(text) <= 3:
        return False

    # v7.12: 整体 "无目标" 模式 → 任何类别都 False
    if "无目标" in text or "无任何目标" in text or "未检测到任何" in text or "0 个" in text:
        return False

    # 找到 cls 出现的位置
    positions = [m.start() for m in re.finditer(cls, text)]
    if not positions: return False  # cls 完全不在 text → 默认 False (保守, 没提就是没)
    # v7.8: 简化为字符级检测
    # 强 negative (整词匹配, 优先级高)
    negative_kw = ["没有", "未检测到", "不存在", "0 个", "否", "无任何", "无目标", "图中无"]
    # 强 positive (有 detection 必须字符级, 避免 "没有汽车" 含 "有汽车" 误判)
    positive_kw_phrase = ["检测到", "1 个", "2 个", "是一", "一辆", "一个"]

    has_positive = False
    has_negative = False
    for pos in positions:
        # 检查 cls 前面 7 字符
        before_ctx = text[max(0, pos-7):pos]
        # cls 后面 7 字符
        after_ctx = text[pos+len(cls):min(len(text), pos+len(cls)+7)]
        full_ctx = before_ctx + text[pos:pos+len(cls)] + after_ctx

        # negative 强匹配 (整词)
        for nw in negative_kw:
            if nw in full_ctx or nw in text[max(0, pos-3):pos]:
                has_negative = True

        # positive 强匹配 (整词)
        for pw in positive_kw_phrase:
            if pw in full_ctx:
                has_positive = True

        # 字符级 "有" 检测: 在 cls 前面 5 字符内, 但排除 "没有"/"无有"/"未有"
        # 而且只在 full_ctx 不含 negative kw 时才计 (避免 "不存在X" 误判)
        if not has_negative:
            has_pos_kw_only_yu = False
            for idx, ch in enumerate(before_ctx):
                if ch == "有":
                    before_1 = before_ctx[max(0, idx-1):idx]
                    before_2 = before_ctx[max(0, idx-2):idx]
                    if before_2 != "没" and before_1 != "未" and before_1 != "无":
                        has_pos_kw_only_yu = True
            if has_pos_kw_only_yu:
                has_positive = True
    if has_negative and not has_positive: return False
    if has_positive and not has_negative: return True
    return None
    if has_negative and not has_positive: return False
    if has_positive and not has_negative: return True
    return None


def parse_bbox_nums(text: str) -> Optional[List[Tuple[int, int]]]:
    """从文本提取 bbox 数字, 返回 [(row_min, row_max), (col_min, col_max)]."""
    # 标准格式
    m = re.search(
        r'row_min\s*=\s*(\d+)\s*,\s*col_min\s*=\s*(\d+)\s*,\s*row_max\s*=\s*(\d+)\s*,\s*col_max\s*=\s*(\d+)',
        text, re.IGNORECASE
    )
    if m:
        return [(int(m.group(1)), int(m.group(3))), (int(m.group(2)), int(m.group(4)))]
    # 中文格式: 行 X-Y、列 Z-W
    m = re.search(r'行\s*(\d+)\s*[-~到至]\s*(\d+)\s*[,，、]\s*列\s*(\d+)\s*[-~到至]\s*(\d+)', text)
    if m:
        return [(int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))]
    # 简化: 行 X-Y 列 Z-W
    m = re.search(r'行\s*(\d+)\s*[-~]\s*(\d+).{0,20}列\s*(\d+)\s*[-~]\s*(\d+)', text)
    if m:
        return [(int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))]
    return None


def is_valid_bbox(bbox, view: str) -> bool:
    """bbox 在视图范围内, 严格闭区间, 面积>0."""
    if bbox is None or len(bbox) != 2: return False
    if view not in VIEW_SHAPES: return False
    H, W = VIEW_SHAPES[view]
    (rmin, rmax), (cmin, cmax) = bbox
    if not (0 <= rmin < rmax < H): return False  # 严格
    if not (0 <= cmin < cmax < W): return False
    if (rmax - rmin) * (cmax - cmin) <= 0: return False
    return True


def parse_numbers(text: str) -> List[float]:
    nums = []
    for m in re.finditer(r"(-?\d+\.?\d*)", text):
        try:
            nums.append(float(m.group(1)))
        except ValueError:
            pass
    return nums


def parse_ranges(text: str) -> List[Tuple[float, float]]:
    ranges = []
    for m in re.finditer(r"(-?\d+\.?\d*)\s*[-到至~]\s*(-?\d+\.?\d*)", text):
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) < 1000 and abs(b) < 1000:
            ranges.append((min(a, b), max(a, b)))
    return ranges


def parse_first_class_answer(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else ""


def parse_target_count(text: str) -> int:
    m = re.search(r"(\d+)\s*个\s*目标", text)
    if m: return int(m.group(1))
    if re.search(r"没有\s*目标|无目标|没有\s*任何\s*目标|不存在", text): return 0
    if re.search(r"检测到\s*(\d+)", text): return int(re.search(r"检测到\s*(\d+)", text).group(1))
    return -1


def parse_has_any_target(text: str) -> Optional[bool]:
    """True=有任意目标, False=无目标, None=模糊."""
    has_kw = ["检测到", "有目标", "存在", "目标1:", "图中检测到", "共 ", "1 个", "2 个"]
    no_kw = ["无目标", "没有目标", "没有检测到", "未检测到", "不存在", "0 个"]
    has_c = sum(1 for k in has_kw if k in text)
    no_c = sum(1 for k in no_kw if k in text)
    if no_c > has_c: return False
    if has_c > no_c: return True
    return None


def bbox_iou(b1, b2) -> float:
    """计算两个 bbox 的 IoU. b1, b2 = ((rmin, rmax), (cmin, cmax))."""
    (r1min, r1max), (c1min, c1max) = b1
    (r2min, r2max), (c2min, c2max) = b2
    inter_rmin = max(r1min, r2min); inter_rmax = min(r1max, r2max)
    inter_cmin = max(c1min, c2min); inter_cmax = min(c1max, c2max)
    if inter_rmax <= inter_rmin or inter_cmax <= inter_cmin: return 0.0
    inter = (inter_rmax - inter_rmin) * (inter_cmax - inter_cmin)
    area1 = max(0, r1max - r1min) * max(0, c1max - c1min)
    area2 = max(0, r2max - r2min) * max(0, c2max - c2min)
    union = area1 + area2 - inter
    if union <= 0: return 0.0
    return inter / union


def number_match_score_strict(ans_nums: List[float], gt_nums: List[float], tol=0.05) -> float:
    """数值严格匹配 (≤5% 容差). 用户要求."""
    if not gt_nums: return 1.0 if not ans_nums else 0.5
    if not ans_nums: return 0.0
    matched = 0
    for g in gt_nums:
        for a in ans_nums:
            if abs(g) < 0.01:
                if abs(a - g) < 0.05: matched += 1; break
            elif abs(a - g) / max(abs(g), 0.01) < tol:
                matched += 1; break
    return matched / len(gt_nums)


def range_match_score(ans_ranges, gt_ranges, tol=0.10) -> float:
    if not gt_ranges: return 1.0 if not ans_ranges else 0.5
    if not ans_ranges: return 0.0
    matched = 0
    for gr in gt_ranges:
        for ar in ans_ranges:
            inter_lo, inter_hi = max(gr[0], ar[0]), min(gr[1], ar[1])
            if inter_hi <= inter_lo: continue
            inter = inter_hi - inter_lo
            union = max(gr[1], ar[1]) - min(gr[0], ar[0])
            iou = inter / union if union > 0 else 0
            if iou > 0.5:
                matched += 1; break
    return matched / len(gt_ranges)


def is_refusal(text: str) -> bool:
    """模型是否在 refusal (说"请提供"等)."""
    refusal_kw = ["请提供", "未提供", "没有提供", "无法分析", "不能判断",
                 "请检查您的输入", "我没有看到具体"]
    return any(k in text for k in refusal_kw)


def eval_qa_pair(question: str, true_answer: str, gen_answer: str) -> Dict:
    """v3 评估: 严格分 ask-class presence vs any-target presence.

    关键指标:
    - asked_class_presence_correct: 问的类别是否准确判断有无
    - any_target_correct: 整图是否有目标判断对
    - false_negative_penalty: 漏检大惩罚 (any_target=有, 但 model 答无目标)
    - class_match: 类别选对 (A/B/C)
    - count_match: 数量对
    - view_match: 视图对应
    - bbox_match: bbox 在范围内 + 数值 ±10 像素
    - num_match_score_strict: 数值 ≤5% 容差
    - refusal: 单独标记
    - bbox_invalid: bbox 越界
    """
    metrics = {}

    # === 0. Refusal 检查 ===
    metrics['refusal'] = int(is_refusal(gen_answer))

    # === 1. 视图一致性 ===
    q_view = detect_view(question)
    g_view = detect_view(gen_answer)
    if q_view is not None:
        metrics['q_view'] = q_view
        if g_view is not None:
            metrics['g_view'] = g_view
            metrics['view_match'] = int(q_view == g_view)
        else:
            metrics['g_view'] = None
            metrics['view_match'] = 0
    view_penalty = 1.0 if metrics.get('view_match', 1) == 1 else 0.0

    # === 2. Ask-class presence (用户要求) ===
    # 例: 问"是否有行人", GT "没有", model 答"图中无目标" →
    #     asked_class 应该是 "行人", asked_class_presence_correct=1 (答对)
    #     any_target_correct=0 (整图描述错)
    asked_class = detect_asked_class(question)
    if asked_class is not None:
        metrics['asked_class'] = asked_class
        # GT: 此类别是否存在 (考虑上下文否定)
        gt_has_class = _has_class_in_text(true_answer, asked_class)
        # GEN: 此类别是否被提及
        gen_has_class = _has_class_in_text(gen_answer, asked_class)
        # 答 "无目标" 隐含 "无任何类别" → False
        if "无目标" in gen_answer or "没有目标" in gen_answer or "未检测到" in gen_answer:
            if gen_has_class is None or gen_has_class is False:
                gen_has_class = False
        if gt_has_class is not None and gen_has_class is not None:
            # 两边都能判断: 1=对, 0=错
            metrics['asked_class_gt'] = gt_has_class
            metrics['asked_class_gen'] = gen_has_class
            metrics['asked_class_presence_correct'] = int(gt_has_class == gen_has_class)
        elif gt_has_class is None and gen_has_class is None:
            # 两边都 None → 不计入 (极端 ambiguous)
            metrics['asked_class_presence_correct'] = None
        else:
            # 一边能判断一边 None → 给低惩罚 (0.5 分)
            # (用户建议: None 也计分, 但惩罚因子低)
            metrics['asked_class_gt'] = gt_has_class
            metrics['asked_class_gen'] = gen_has_class
            metrics['asked_class_presence_correct'] = 0.5

    # === 3. Any-target presence (整图是否有目标) ===
    gt_any = parse_has_any_target(true_answer)
    gen_any = parse_has_any_target(gen_answer)
    if gt_any is not None and gen_any is not None:
        metrics['any_target_gt'] = gt_any
        metrics['any_target_gen'] = gen_any
        metrics['any_target_correct'] = int(gt_any == gen_any)
        # 用户要求: 漏检大惩罚
        if gt_any and not gen_any:
            metrics['false_negative_penalty'] = 1  # 漏检

    # === 4. 类别判断 (A/B/C 选对) ===
    if "是什么类别" in question or ("类别" in question and "A." in question):
        gt_letter = parse_first_class_answer(true_answer)
        gen_letter = parse_first_class_answer(gen_answer)
        if gt_letter and gen_letter:
            metrics["class_match"] = int(gt_letter == gen_letter)
            metrics["gt_letter"] = gt_letter
            metrics["gen_letter"] = gen_letter

    # === 5. 数量判断 ===
    if any(k in question for k in ["几个目标", "有行人", "有汽车", "目标数量"]):
        gt_cnt = parse_target_count(true_answer)
        gen_cnt = parse_target_count(gen_answer)
        if gt_cnt >= 0 and gen_cnt >= 0:
            metrics["count_match"] = int(gt_cnt == gen_cnt)
            metrics["count_diff"] = abs(gt_cnt - gen_cnt)

    # === 6. 数值精度 (≤5% 严格) ===
    if any(k in question for k in ["距离", "速度", "多普勒", "角度", "范围", "米", "m/s"]):
        gt_nums = parse_numbers(true_answer)
        gen_nums = parse_numbers(gen_answer)
        metrics["num_match_score"] = number_match_score_strict(gen_nums, gt_nums, tol=0.10) * view_penalty  # v7: 5%→10% 物理量容差更现实
        gt_ranges = parse_ranges(true_answer)
        gen_ranges = parse_ranges(gen_answer)
        metrics["range_match_score"] = range_match_score(gen_ranges, gt_ranges, tol=0.10) * view_penalty

    # === 7. bbox (严格越界检查) ===
    if "边界框" in question or "bbox" in question.lower():
        if metrics.get('view_match', 1) == 0:
            metrics['bbox_match'] = 0
            metrics['bbox_invalid'] = 1
            metrics['bbox_format'] = 'view_mismatch'
        else:
            gt_bbox = parse_bbox_nums(true_answer)
            gen_bbox = parse_bbox_nums(gen_answer)
            metrics['gt_bbox'] = gt_bbox
            metrics['gen_bbox'] = gen_bbox
            # 越界检查
            if gen_bbox is not None and g_view is not None:
                metrics['bbox_invalid'] = int(not is_valid_bbox(gen_bbox, g_view))
            else:
                metrics['bbox_invalid'] = 1
            metrics['bbox_format'] = 'standard' if (gt_bbox and re.search(r'row_min', true_answer)) else 'natural'
            if gt_bbox is None or gen_bbox is None:
                metrics['bbox_match'] = None
            else:
                # 用 IoU 严格评估 bbox (用户要求 IoU)
                iou = bbox_iou(gen_bbox, gt_bbox)
                metrics['bbox_iou'] = iou
                # IoU > 0.5 算对
                metrics['bbox_match'] = int(iou > 0.5)
                # 越界 bbox (用整图 [0, H]×[0, W]) IoU 接近 0, 仍判 0
                # 但 model 故意答 "全图" bbox (IoU=0.5-1.0 全图) 走捷径, 也算错 (iou 实际 < 0.5)
            # 用户要求: 越界不当作正确
            if metrics.get('bbox_invalid', 0) == 1:
                metrics['bbox_match'] = 0
                metrics['bbox_invalid_match'] = 0  # 走捷径答全图, 严格大惩罚

    # === 8. 描述题数值检查 (用户要求: GEN 中任何数字 vs GT, 偏差 > 50% 判错) ===
    is_description_q = any(k in question for k in ["描述", "链式推理", "距离", "速度", "多普勒"])
    if is_description_q and not any(k in question for k in ["几个目标", "是什么类别", "边界框", "bbox"]):
        gt_nums_desc = parse_numbers(true_answer)
        gen_nums_desc = parse_numbers(gen_answer)
        if gt_nums_desc and gen_nums_desc:
            # 每个 GEN 数字, 找 GT 中最近的, 偏差 > 50% 判错
            desc_error = False
            worst_deviation = 0.0
            for g in gen_nums_desc:
                gv = abs(g)
                if gv < 0.01: continue
                closest_gt = min(gt_nums_desc, key=lambda x: abs(x - g))
                if abs(closest_gt) < 0.01: continue
                deviation = abs(gv - abs(closest_gt)) / abs(closest_gt)
                if deviation > 0.5:
                    desc_error = True
                    worst_deviation = max(worst_deviation, deviation)
            metrics['desc_num_valid'] = int(not desc_error)
            if desc_error:
                metrics['desc_worst_deviation'] = worst_deviation

    # === 9. token overlap (降权, 仅参考) ===
    gt_tokens = set(true_answer.replace(" ", ""))
    gen_tokens = set(gen_answer.replace(" ", ""))
    if gt_tokens:
        metrics["token_overlap"] = len(gt_tokens & gen_tokens) / len(gt_tokens)

    return metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """聚合多维度指标."""
    agg = {}
    for k in [
        "class_match", "asked_class_presence_correct", "any_target_correct",
        "false_negative_penalty", "view_match", "count_match",
        "num_match_score", "range_match_score", "bbox_match",
        "refusal", "bbox_invalid", "desc_num_valid", "token_overlap",
    ]:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if vals:
            agg[f"avg_{k}"] = sum(vals) / len(vals)
            agg[f"n_{k}"] = len(vals)
    # 特殊统计
    n_view_mismatch = sum(1 for m in metrics_list if m.get('view_match') == 0)
    n_false_negative = sum(1 for m in metrics_list if m.get('false_negative_penalty') == 1)
    n_bbox_invalid = sum(1 for m in metrics_list if m.get('bbox_invalid') == 1)
    n_refusal = sum(1 for m in metrics_list if m.get('refusal') == 1)
    if n_view_mismatch: agg['n_view_mismatch'] = n_view_mismatch
    if n_false_negative: agg['n_false_negative'] = n_false_negative
    if n_bbox_invalid: agg['n_bbox_invalid'] = n_bbox_invalid
    if n_refusal: agg['n_refusal'] = n_refusal
    agg["n_samples"] = len(metrics_list)
    return agg


if __name__ == "__main__":
    # Test 1: ask-class vs any-target 拆分
    m1 = eval_qa_pair(
        "这张雷达图中是否有行人?",
        "没有, 图中不存在行人。",  # GT: 行人没有 (但图可能没目标)
        "图中无目标。",  # GEN 答无目标
    )
    print("Test 1: 问行人 vs 答无目标")
    print(f"  asked_class=行人 gt_present={m1.get('asked_class_gt')} gen_present={m1.get('asked_class_gen')} → correct={m1.get('asked_class_presence_correct')}")
    print(f"  any_target gt={m1.get('any_target_gt')} gen={m1.get('any_target_gen')} → correct={m1.get('any_target_correct')}")
    # 期望: asked_class correct=1 (对, 行人确实没), any_target correct=0 (错, 应说"没行人"而不是"没目标")
    print()

    # Test 2: 数值严格 (5%)
    m2 = eval_qa_pair(
        "距离是多少?",
        "距离 10 m",
        "距离 10.5 m",  # 5% 偏差, 应该判错
    )
    print(f"Test 2: 10.5 vs 10 (5% 偏差): num_match_score = {m2.get('num_match_score')}")
    print(f"  期望: < 1.0 (因为 10.5 偏差 5%)")
    print()

    # Test 3: 数值 6% 偏差
    m3 = eval_qa_pair(
        "距离是多少?",
        "距离 10 m",
        "距离 10.6 m",  # 6% 偏差, 应该判错
    )
    print(f"Test 3: 10.6 vs 10 (6% 偏差): num_match_score = {m3.get('num_match_score')}")
    print(f"  期望: < 1.0")

    # Test 4: bbox 越界 [-1, -1, 257, 63]
    m4 = eval_qa_pair(
        "请告诉我第一个 汽车 在 RD 视图中的边界框坐标。",
        "RD 视图 bbox: [row_min=80, col_min=40, row_max=85, col_max=41]。",
        "RD 视图 bbox: [row_min=-1, col_min=-1, row_max=257, col_max=63]。",  # 越界
    )
    print(f"Test 4: bbox [-1, -1, 257, 63]: bbox_match={m4.get('bbox_match')}, invalid={m4.get('bbox_invalid')}")
    print(f"  期望: bbox_match=0 (越界)")

    # Test 5: 描述题 100m 幻觉
    m5 = eval_qa_pair(
        "请描述这张雷达图像中检测到的所有目标",
        "距离约 35 m, 速度约 3 m/s",
        "距离 100 m, 速度 50 m/s",  # 幻觉
    )
    print(f"Test 5: 描述 100m/50m/s vs 35m/3m/s: desc_num_valid={m5.get('desc_num_valid')}")
    print(f"  期望: desc_num_valid=0 (偏差 > 50%)")