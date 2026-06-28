# Student KD Architecture

The retained student is the compact KD model. Teacher and student prediction
heads are separate, but deployment cost depends on how the student inputs are
produced. The `visual_text_sound` preset consumes teacher-frontend artifacts.
The `clip_mobilenet_text` preset replaces the visual frontend with CLIP and
MobileNet, while the current experiment still consumes cached `text_pooled`.
Therefore, head-only parameter counts are reported for reproducibility but are
not used as end-to-end compression ratios.

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
does not use action/caption auxiliary prediction heads. Recreating
`text_pooled` from raw inputs requires the YAMNet sound classifier and Stable
Diffusion v1.x CLIP text encoder; those modules are included in the raw-input
parameter audit even though the experiments read their cached outputs.

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

| Model | Training objective | PLCC | SRCC | Final | MSE | MAE | Student checkpoint / ECR path | Raw-input E2E params | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official teacher | Released checkpoint; no thesis training | 0.7103 | 0.6995 | **0.7038** | — | — | N/A | ~1,801.70M | — |
| Student KD (full) | Ground truth + teacher score/clip/feature/attention/ranking/relation losses | 0.6304 | 0.6252 | **0.6273** | 0.0562 | 0.1768 | 433,092 / **378,723** | ~274.11M | 35 |
| Student KD basic | **Only** `MSE(student_ecr, teacher_ecr)` | 0.6277 | 0.6193 | **0.6227** | 0.0546 | 0.1758 | 433,092 / **378,723** | ~274.11M | 32 |
| Student No KD | **Only** `MSE(student_ecr, true_ecr)` | 0.5634 | 0.5593 | **0.5609** | 0.0661 | 0.1915 | 433,092 / **378,723** | ~274.11M | 5 |
| Proper KD (`clip_mobilenet_text`) | Full KD objective with independent visual extractors | 0.5798 | 0.5743 | **0.5765** | 0.0636 | 0.1888 | 1,821,060 / **1,530,435** | ~217.12M | 28 |

`Student checkpoint / ECR path` is retained for exact checkpoint reproducibility. Compression claims must use `Raw-input E2E params`, which include all neural feature extractors needed to recreate the model inputs. The semi-independent total includes EfficientNetV2-S, UVQ Distortion, EVQA frame-fusion layers, CLIP ViT-B/32 visual features, the Stable Diffusion v1.x CLIP text encoder, YAMNet, and the deployed student path. The Proper KD total replaces the teacher frame frontend with MobileNetV3-Small, but still includes the text encoder and YAMNet required to recreate the cached `text_pooled` input. See the component-level audit in the root README.
