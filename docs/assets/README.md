# docs/assets/

This directory holds the project **visual assets** (architecture diagrams, demo screenshots, etc.).

When you run the demo, take a screenshot of the browser at `http://localhost:8765` and drop it here as `demo_screenshot.png` to populate the README. Suggested screenshots:

| File | What it shows |
| --- | --- |
| `demo_screenshot.png` | The full demo page: 3 RD/RA/AD view canvas + chat panel + GT vs PKC legend |
| `chat_example.png` | A finished chat session with several QA rounds (close-up of right panel) |
| `pkc_vs_gt.png` | Single-frame close-up showing GT (red) and PKC (yellow dashed) overlays |

ASCII fallback (so the README is still useful without images):

```
┌────────────────────────────────────────────────────────────────┐
│  RD 视图  (256×64)  │  RA 视图  (256×256)  │  AD 视图  (256×64)  │
│  ┌──────────────┐   │  ┌──────────────┐   │  ┌──────────────┐   │
│  │              │   │  │              │   │  │              │   │
│  │   ●red GT   │   │  │              │   │  │              │   │
│  │   ┄┄yellow PKC   │  │              │   │  │              │   │
│  └──────────────┘   │  └──────────────┘   │  └──────────────┘   │
├────────────────────────────────────────────────────────────────┤
│  PKC 检测: 汽车 (car) range=9.2m angle=-2.8° doppler=9.7m/s    │
│  GT 标注: 1 instance [RD bbox 206,53 → 210,56]                │
│                                                                │
│  Chat:                                                          │
│   You: 距离?              Bot: 9.2m                             │
│   You: 类别?              Bot: 汽车                              │
│   You: 有骑行者吗?       Bot: 无                                │
└────────────────────────────────────────────────────────────────┘
```

The PNG files are not committed by default because of the `.gitignore` rules — add them via `git add -f` when you have them.
