# Data directory

This directory is reserved for **PKC inference helpers** used at training/eval time.

The CARRADA dataset itself is **not** redistributed in this repo. Download it from the official source and pre-process it with the CARRADA-provided scripts into `.npy` spectrograms:

```
/data/storage/zzy/Carrada/
├── 2019-09-16-12-52-12/
│   ├── range_doppler_processed/
│   │   ├── 000000.npy
│   │   └── ...
│   ├── range_angle_processed/
│   └── angle_doppler_processed/
└── annotations_frame_oriented.json
```

Default paths used in the code (override via CLI flag if yours differ):

| What | Path |
| --- | --- |
| CARRADA root | `/data/storage/zzy/Carrada` |
| Pre-trained PKC weights | `radarlm/pkc_backbone/weights/pkcin_silu_gn.pt` |
| Generated v12 data | `/data/storage/zzy/radar_agent_data/{train,val,test}_qwen_mt_v12.jsonl` |
| Trained model output | `output/v9_qa_ddp_v12/` |

## Generating the v12 curriculum dataset

After you have CARRADA in the expected layout, run:

```bash
python radarlm/vlm/generate_v12_curriculum.py
```

This will:
1. Run PKC on all 12,666 frames → store `object list` per frame
2. Build 3-stage curriculum prompts (Stage 1 = full list, Stage 2 = partial, Stage 3 = image only)
3. Save 96,870 train / 15,860 val / 13,930 test records

Expected runtime: ~7 min on 1×RTX 4090.

> If your paths differ, edit the `CARRADA_ROOT` and `OUTPUT_*` constants at the top of `radarlm/vlm/generate_v12_curriculum.py`.
