# Student KD Architecture

The retained student is the compact KD model. Teacher and student are separate
networks: the teacher artifacts supervise training, while deployment runs only
the lightweight student-side inputs and model.

## Student Input And Forward Path

```text
frame_fusion_feature (T x 1024) ─┐
CLIP keyframe feature (T x 512) ─┴─ clip_add projection
                                  │
                                  ▼
                    Temporal Transformer
                    (1 layer, 4 heads, 96d)
                                  │
                           attention pooling
                                  │
sound/title/description           │
text_pooled (3 x 768)             │
        │                         │
        ▼                         │
text projection + source embedding
        │                         │
        └────────── multimodal fusion
                             │
                         ECR head
                             │
                       student_ecr
```

`text_pooled` contains only sound classification, title, and description.
Generated-caption text is not a deployed student input. The retained student
does not use action/caption auxiliary prediction heads.

## Retained Teacher Artifacts

| Artifact | Direct student input | Training-only KD target |
|---|:---:|:---:|
| `frame_fusion_feature` | Yes | No |
| `text_pooled` (sound/title/description) | Yes | No |
| `teacher_ecr`, `clip_ecr` | No | Yes |
| `fusion_hidden`, `temporal_hidden` | No | Yes |
| `attention_importance` | No | Yes |
Raw `text_tokens`, generated-caption pooled text, full attention matrices,
`action_feature`, `caption_feature`, and caption JSONL sidecars are not consumed
by the retained student.

## KD Objective

The full objective combines ground-truth ECR, teacher score and clip outputs,
hidden-feature alignment, temporal attention, ranking, and relation losses.
No-KD and basic-KD use the same controlled input and student architecture; only
their objectives differ.

## Experimental Results

The controlled loss ablation uses the same `visual_text_sound` input, CLIP
`clip_add`, 96 hidden dimensions, one Transformer layer, seed 42, and the same
deterministic 4000/1000 split.

| Model | Training objective | PLCC | SRCC | Final | MSE | MAE | Train params | Inference params (current / pruned) | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official teacher | Released checkpoint; no thesis training | 0.7103 | 0.6995 | **0.7038** | — | — | N/A | ~1,801.7M / same | — |
| Student KD (full) | Ground truth + teacher score/clip/feature/attention/ranking/relation losses | 0.6304 | 0.6252 | **0.6273** | 0.0562 | 0.1768 | 433,092 | 433,092 / **378,723** | 35 |
| Student KD basic | **Only** `MSE(student_ecr, teacher_ecr)` | 0.6277 | 0.6193 | **0.6227** | 0.0546 | 0.1758 | 433,092 | 433,092 / **378,723** | 32 |
| Student No KD | **Only** `MSE(student_ecr, true_ecr)` | 0.5634 | 0.5593 | **0.5609** | 0.0661 | 0.1915 | 433,092 | 433,092 / **378,723** | 5 |
