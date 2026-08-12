"""收集 v4 模型的 badcase 表格."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

import torch
from transformers import AutoTokenizer
from radarlm.vlm.train_v9_qa_ddp import PKCQwenAlign, RadarQADataset
from radarlm.vlm.eval_v9_v2_runner import build_single_qa_prompt
from radarlm.vlm.eval_v9_v2 import eval_qa_pair


def classify_qa(question, gen, gt):
    """简单把 QA 分类."""
    q = question.lower()
    if "bbox" in q.lower() or "边界框" in q:
        return "bbox"
    if "是什么类别" in q or "类别" in q and "A." in q:
        return "类别"
    if "几个目标" in q:
        return "数量"
    if "距离" in q or "多普勒" in q or "速度" in q or "角度" in q or "m/s" in q:
        return "数值"
    if "RD/RA/AD" in q or "RD(距离-多普勒)" in q or "三个视图" in q:
        return "三视图"
    if "为什么" in q:
        return "描述"
    if "描述" in q or "链式推理" in q:
        return "描述"
    return "其他"


def main():
    print("加载 v4 模型...")
    ds = RadarQADataset("/data/storage/zzy/radar_agent_data/test_qwen_mt.jsonl",
                        "/data/storage/zzy/Carrada", seed=42)
    model = PKCQwenAlign("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct").cuda()
    sd = torch.load("/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_v4/projector_e3.pt", weights_only=False)
    model.proj_rd.load_state_dict(sd["proj_rd"])
    model.proj_ra.load_state_dict(sd["proj_ra"])
    model.eval()
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")

    badcase_records = []
    n_samples = 30
    print(f"生成 {n_samples} samples × 5 QA badcase...")
    for i in range(n_samples):
        sample = ds[i]
        x_rd = sample["x_rd"].cuda()
        x_ra = sample["x_ra"].cuda()
        x_ad = sample["x_ad"].cuda()
        visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
        image_grid_thw = torch.tensor([[1, 32, 64]], dtype=torch.long).cuda()

        qa_pairs = []
        for j in range(len(sample["conversations"]) - 1):
            if sample["conversations"][j]["from"] == "user":
                q = sample["conversations"][j]["value"].replace("<image>", "").strip()
                if j + 1 < len(sample["conversations"]):
                    a = sample["conversations"][j + 1]["value"]
                    qa_pairs.append((q, a))

        for k, (q, gt_a) in enumerate(qa_pairs):
            prompt = build_single_qa_prompt(sample["conversations"], k, 320)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()
            with torch.no_grad():
                out_ids = model.qwen.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=visual_embeds, image_grid_thw=image_grid_thw,
                    min_new_tokens=20, max_new_tokens=128, do_sample=False, repetition_penalty=1.1,
                )
            gen_a = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
            m = eval_qa_pair(q, gt_a, gen_a)
            qa_type = classify_qa(q, gen_a, gt_a)
            # 各种 metric 状态
            view_match = m.get("view_match", "N/A")
            num_match = m.get("num_match_score", "N/A")
            count_match = m.get("count_match", "N/A")
            has_target = m.get("has_target_match", "N/A")
            class_match = m.get("class_match", "N/A")
            # 判定答错: 任何一个关键 metric=0
            wrong = False
            reasons = []
            if class_match == 0:
                wrong = True; reasons.append("类别错")
            if count_match == 0:
                wrong = True; reasons.append("数量错")
            if view_match == 0:
                wrong = True; reasons.append("视图错")
            if num_match == 0 and qa_type in ["数值", "bbox"]:
                wrong = True; reasons.append("数值错")
            if has_target == 0:
                wrong = True; reasons.append("有无错")
            # 答含"无目标" 字样
            if "无目标" in gen_a and "无目标" not in gt_a:
                wrong = True
                reasons.append("答无目标但实际有")
            if "无目标" in gt_a and "无目标" not in gen_a:
                wrong = True
                reasons.append("答有目标但实际无")
            badcase_records.append({
                "sample_id": i,
                "qa_idx": k,
                "qa_type": qa_type,
                "question": q[:80],
                "gt_answer": gt_a[:150],
                "gen_answer": gen_a[:150],
                "wrong": wrong,
                "reasons": reasons,
                "view_match": view_match,
                "num_match": num_match,
                "count_match": count_match,
                "has_target": has_target,
                "class_match": class_match,
            })

    # 写入 JSON
    out_path = "/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_v4/badcases.json"
    with open(out_path, "w") as f:
        json.dump(badcase_records, f, indent=2, ensure_ascii=False)
    print(f"保存 {len(badcase_records)} QA 到 {out_path}")

    # 打印典型 badcase
    print("\n=== 典型 Badcase 样本 (前 15 个错) ===")
    printed = 0
    for r in badcase_records:
        if printed >= 15: break
        if not r["wrong"]: continue
        print(f"\n--- 样本 {r['sample_id']} QA {r['qa_idx']} ({r['qa_type']}) ---")
        print(f"Q: {r['question']}")
        print(f"TRUE: {r['gt_answer']}")
        print(f"GEN:  {r['gen_answer']}")
        print(f"错: {', '.join(r['reasons'])}")
        print(f"metrics: view={r['view_match']} num={r['num_match']} count={r['count_match']} has={r['has_target']} class={r['class_match']}")
        printed += 1


if __name__ == "__main__":
    main()