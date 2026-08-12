"""v12 curriculum learning 数据生成器
1. 用 PKC 推理 12666 个 frame → x9 logits
2. 解码出 object list
3. 按 Stage 1/2/3 生成 QA
4. 写入 train_qwen_mt_v12.jsonl
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

PROJECT_DIR = Path("/home/zzy/Myproject/RadarLM")
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.pkc_decoder import decode_pkc_segmentation, objects_to_text

CARRADA_ROOT = "/data/storage/zzy/Carrada"
ANNOTATIONS_PATH = f"{CARRADA_ROOT}/annotations_frame_oriented.json"
OUTPUT_TRAIN = "/data/storage/zzy/radar_agent_data/train_qwen_mt_v12.jsonl"
OUTPUT_VAL = "/data/storage/zzy/radar_agent_data/val_qwen_mt_v12.jsonl"
OUTPUT_TEST = "/data/storage/zzy/radar_agent_data/test_qwen_mt_v12.jsonl"

# PKC_NORM_STATS (与 train_v9_qa_ddp.py 一致)
PKC_NORM_STATS = {
    "rd": (37.59535773996415, 119.08313902425246),
    "ra": (40.40928894952408, 103.80548746494114),
    "ad": (54.42604354196056, 105.79746676271202),
}


def load_npy_from_carrada(seq, frame, view):
    name_map = {"rd": "range_doppler_processed", "ra": "range_angle_processed", "ad": "angle_doppler_processed"}
    p = Path(CARRADA_ROOT) / seq / name_map[view] / f"{frame}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def center_crop(t, h_out, w_out):
    if t is None:
        return None
    H, W = t.shape[:2]
    if H < h_out or W < w_out:
        t = np.pad(t, ((0, max(0, h_out - H)), (0, max(0, w_out - W))), mode='constant')
        H, W = t.shape[:2]
    return t[(H - h_out) // 2:(H - h_out) // 2 + h_out, (W - w_out) // 2:(W - w_out) // 2 + w_out]


def normalize_view(arr, view):
    min_v, max_v = PKC_NORM_STATS[view]
    return np.clip((arr.astype(np.float32) - min_v) / (max_v - min_v), 0.0, 1.0)

# 物理量映射
ranges_per_class = {1: 'near', 2: 'near', 3: 'near', 4: 'mid', 5: 'far',
                    6: 'far', 7: 'far', 8: 'far', 9: 'far'}


def gen_qa_for_objects(objects_dict, qa_type, target_class=None):
    """从 object list 生成单个 QA 答案.
    qa_type: 与 QA_TEMPLATES 中的 qa_prompt 完全一致 (含 ? 但 split 之后), e.g. "图中是否有目标？"
    """
    objs = objects_dict.get('objects', [])
    has_target = len(objs) > 0

    # 过滤指定类别
    if target_class is not None:
        filtered = [o for o in objs if o['class'] == target_class]
    else:
        filtered = objs

    # 答案表 (key 与 QA_TEMPLATES 中 qa_prompt split("？")[0].strip() 完全一致)
    if not has_target:
        no_target = "图中无目标"
        answers = {
            "图中是否有目标": no_target,
            "图中有几个目标": no_target,
            "图中目标是什么类别": no_target,
            "最近目标的距离是多少": no_target,
            "最近目标的角度是多少": no_target,
            "最近目标的多普勒速度是多少": no_target,
            "图中有几个汽车": no_target,
            "图中有几个行人": no_target,
            "图中有骑行者吗": no_target,
            "图中有汽车吗": no_target,
        }
        return answers.get(qa_type, no_target)

    main_obj = filtered[0] if filtered else objs[0]

    matchers = {
        "图中是否有目标": "有" if has_target else "无",
        "图中有几个目标": str(len(objs)),
        "图中目标是什么类别": ", ".join(o['class_cn'] for o in objs),
        "最近目标的距离是多少": f"{main_obj['range_m']:.1f}m",
        "最近目标的角度是多少": f"{main_obj['angle_deg']:.1f}°",
        "最近目标的多普勒速度是多少": f"{main_obj['doppler_ms']:.1f}m/s",
        "图中有几个汽车": str(sum(1 for o in objs if o['class'] == 3)),
        "图中有几个行人": str(sum(1 for o in objs if o['class'] == 1)),
        "图中有骑行者吗": "有" if any(o['class'] == 2 for o in objs) else "无",
        "图中有汽车吗": "是" if any(o['class'] == 3 for o in objs) else "否",
    }
    return matchers.get(qa_type, "未知")


# QA 模板 (同训练时一致)
QA_TEMPLATES = [
    ("图中是否有目标？", None),
    ("图中有几个目标？", None),
    ("图中目标是什么类别？", None),
    ("最近目标的距离是多少？", None),
    ("最近目标的角度是多少？", None),
    ("最近目标的多普勒速度是多少？", None),
    ("图中有几个汽车？", 3),
    ("图中有几个行人？", 1),
    ("图中有骑行者吗？", 2),
    ("图中有汽车吗？", 3),
]


def get_carrada_frame_list():
    """从 annotations 拿 frame 列表 (按 train/val/test 拆)"""
    ann = json.load(open(ANNOTATIONS_PATH))
    return ann


def build_v12_qa_pair(image_text, qa_prompt, qa_answer, image_pad_str):
    """构造 VLM 训练 prompt (curriculum version)"""
    # image_text 是 Stage 1/2/3 不同 detail 的 object list
    # image_pad_str 始终有 (VLM 还能看图)
    if image_text:
        # Stage 1/2: text + image
        return (f"<|im_start|>system\n你是雷达感知系统. 结合视觉与PKC检测结果回答.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{image_pad_str}{image_text}\n{qa_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{qa_answer}<|im_end|>\n")
    else:
        # Stage 3: 仅图像
        return (f"<|im_start|>system\n你是雷达感知系统. 通过分析雷达图像回答.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{image_pad_str}{qa_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{qa_answer}<|im_end|>\n")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "small":
        MAX_FRAMES = 20  # 限制 frame 数测试
    else:
        MAX_FRAMES = None

    print("[Setup] loading annotations...")
    ann = get_carrada_frame_list()

    print("[Setup] loading PKC model...")
    pkc = PKCWithPretrained(
        n_classes=4, n_frames=5, device='cuda',
        weights_path=str(PROJECT_DIR / "radarlm/pkc_backbone/weights/pkcin_silu_gn.pt")
    )
    pkc.eval()
    pkc_internal = pkc.pkc  # 直接访问内部 PKCIn_plus_cvf_aug, 支持 latent_type
    pkc_internal = pkc_internal.cuda()  # 强制 CUDA

    # 收集 frame
    seqs = sorted(ann.keys())[:30]  # 全 30 个 seq
    frames = []
    for seq in seqs:
        for fid in sorted(ann[seq].keys(), key=lambda x: int(x)):
            frames.append((seq, fid))
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    print(f"[Setup] {len(frames)} frames to process")

    # 跑 PKC 推理
    print("[Run] Generating object lists via PKC...")
    objects_per_frame = {}  # (seq, fid) -> [obj]
    cur = int("0") - 1  # 不用
    with torch.no_grad():
        for seq, fid in tqdm(frames):
            cur = int(fid)
            # 5 帧 (cur + 过去 4)
            frame_ints = [cur - i for i in range(5)]
            rd5, ra5, ad5 = [], [], []
            for f in frame_ints:
                fid_str = f"{f:06d}"
                rd = load_npy_from_carrada(seq, fid_str, "rd")
                ra = load_npy_from_carrada(seq, fid_str, "ra")
                ad = load_npy_from_carrada(seq, fid_str, "ad")
                if rd is None or ra is None or ad is None:
                    rd = np.zeros((256, 64), dtype=np.float32) if rd is None else rd
                    ra = np.zeros((256, 256), dtype=np.float32) if ra is None else ra
                    ad = np.zeros((256, 64), dtype=np.float32) if ad is None else ad
                rd5.append(center_crop(rd, 256, 64))
                ra5.append(center_crop(ra, 256, 256))
                ad5.append(center_crop(ad, 256, 64))
            rd5 = np.stack(rd5); ra5 = np.stack(ra5); ad5 = np.stack(ad5)
            rd5 = normalize_view(rd5, "rd")
            ra5 = normalize_view(ra5, "ra")
            ad5 = normalize_view(ad5, "ad")
            x_rd = torch.from_numpy(rd5).unsqueeze(0).unsqueeze(0).cuda().float()
            x_ra = torch.from_numpy(ra5).unsqueeze(0).unsqueeze(0).cuda().float()
            x_ad = torch.from_numpy(ad5).unsqueeze(0).unsqueeze(0).cuda().float()
            # 确保 dtype 一致 (PKC 用 float32)
            x_rd = x_rd.float()
            x_ra = x_ra.float()
            x_ad = x_ad.float()
            # PKC inference → x9 logits
            x9_rd, x9_ra = pkc_internal(x_rd, x_ra, x_ad, features_only=True, latent_type='x9')
            # x9_rd: (B, 4, 256, 64) → (4, 256, 64)
            x9_rd = x9_rd.squeeze(0).cpu().numpy()
            x9_ra = x9_ra.squeeze(0).cpu().numpy()
            objs = decode_pkc_segmentation(x9_rd, x9_ra, None)
            objects_per_frame[(seq, fid)] = objs

    # 生成 v12 QA
    IMG_PAD = "<|image_pad|>" * 320  # v9 标准 (注意: 这是 320 个 <|image_pad|>, 长度 3840 chars)
    train_data, val_data, test_data = [], [], []
    np.random.seed(42)
    for seq, fid in tqdm(frames):
        seq_idx = seqs.index(seq)
        split = "train" if seq_idx < 18 else ("val" if seq_idx < 24 else "test")
        objs = objects_per_frame[(seq, fid)]
        objs_dict = {"objects": objs}

        for qa_prompt, target_cls in QA_TEMPLATES:
            # qa_type 与 matchers key 完全对齐 (split 掉 ？)
            qa_type = qa_prompt.split("？")[0].strip()
            answer = gen_qa_for_objects(objs_dict, qa_type, target_cls)

            # Stage 1: 完整 object list
            text_1 = objects_to_text(objs, level=1)
            qa_1 = build_v12_qa_pair(text_1, qa_prompt, answer, IMG_PAD)

            # Stage 2: 只类别名
            text_2 = objects_to_text(objs, level=2)
            qa_2 = build_v12_qa_pair(text_2, qa_prompt, answer, IMG_PAD)

            # Stage 3: 仅图像
            text_3 = ""
            qa_3 = build_v12_qa_pair(text_3, qa_prompt, answer, IMG_PAD)

            record = {
                "id": f"{seq}_{fid}",
                "image": IMG_PAD,
                "conversations": [
                    {"from": "user", "value": qa_prompt.replace("<image>", IMG_PAD)},
                    {"from": "assistant", "value": answer + "<|im_end|>"},
                ],
                "stage_1_prompt": qa_1,
                "stage_2_prompt": qa_2,
                "stage_3_prompt": qa_3,
                "n_objects": len(objs),
            }
            if split == "train": train_data.append(record)
            elif split == "val": val_data.append(record)
            else: test_data.append(record)

    print(f"\n[Save] train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    for path, data in [(OUTPUT_TRAIN, train_data), (OUTPUT_VAL, val_data), (OUTPUT_TEST, test_data)]:
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  → {path}")


if __name__ == "__main__":
    main()