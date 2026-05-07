# Official SnapUGC Teacher Source

This directory vendors the official teacher inference architecture used for the
current 5000-video thesis run.

Source repository:

```text
https://github.com/dasongli1/SnapUGC_Engagement.git
```

Pinned commit:

```text
4e0ce3154225cfdf1d036e5b8b1d3874615a04f7
```

Only `ECR_inference/` is kept here. Sample videos, validation CSVs, generated
submissions, and checkpoints are intentionally excluded. Required checkpoints
remain external data and should live under `checkpoints/`, Kaggle input, or the
GCloud checkpoint directory.

The key teacher architecture files are:

```text
ECR_inference/modules/EVQA.py
ECR_inference/test_SnapUGC_baseline.py
ECR_inference/modules/efficientnet_v2.py
ECR_inference/modules/distort.py
ECR_inference/modules/resnet3d.py
ECR_inference/mPLUG_2/models/model_video_caption_mplug2.py
ECR_inference/mPLUG_2/models/visual_transformers.py
```

Runtime scripts may patch a working copy for modern PyTorch/Transformers
compatibility and artifact export. Treat this vendored copy as the clean
reference for what the paper teacher architecture actually is.

## Teacher Architecture

This is the official released EVQA teacher path used by the current GCloud
5000-video run.

```mermaid
flowchart TD
    A["Input video .mp4"] --> B1["RGB frames"]
    A --> B2["Audio waveform"]
    A --> B3["Video frames for mPLUG-2"]

    M["Metadata CSV"] --> T1["Title"]
    M --> T2["Description"]

    B1 --> C1["EfficientNetV2-s<br/>semantic frame feature<br/>528d per frame"]
    B1 --> C2["UVQ / Distortion net<br/>distortion frame feature<br/>256d per frame"]
    B1 --> C3["ResNet3D-18<br/>action clip feature<br/>512d per 16-frame clip"]

    B3 --> C4["mPLUG-2<br/>caption text + clip feature<br/>1024d per clip"]
    B2 --> C5["YAMNet<br/>top-5 sound labels as text"]

    C5 --> D1["Stable Diffusion text encoder<br/>sound tokens 77 x 768"]
    T1 --> D2["Stable Diffusion text encoder<br/>title tokens 77 x 768"]
    T2 --> D3["Stable Diffusion text encoder<br/>description tokens 77 x 768"]
    C4 --> D4["Stable Diffusion text encoder<br/>caption tokens 77 x 768"]

    C1 --> E1["fc1: 528 -> 256"]
    C2 --> E2["fc3: 256 -> 256"]
    E1 --> F1["Concat semantic + distortion<br/>per frame"]
    E2 --> F1
    F1 --> F2["Group every 16 frames"]
    F2 --> F3["fc_merge12<br/>frame fusion feature<br/>1024d per clip"]

    C3 --> G1["feat3_preprocess<br/>512d action query"]
    C3 --> G2["fc30<br/>action feature 512d"]

    C4 --> H1["fc4<br/>mPLUG feature 1024d"]

    G1 --> I1["Cross-attention with sound text<br/>512d"]
    D1 --> I1
    G1 --> I2["Cross-attention with title text<br/>512d"]
    D2 --> I2
    G1 --> I3["Cross-attention with description text<br/>512d"]
    D3 --> I3
    G1 --> I4["Cross-attention with caption text<br/>512d"]
    D4 --> I4

    F3 --> J["Concat all clip-level features<br/>1024 + 512 + 1024 + 4*512 = 4608d"]
    G2 --> J
    H1 --> J
    I1 --> J
    I2 --> J
    I3 --> J
    I4 --> J

    J --> K["fc_merge123<br/>4608 -> 512<br/>fusion_hidden"]
    K --> L["8-layer temporal self-attention Transformer<br/>512d per clip<br/>temporal_hidden"]
    L --> O["Output MLP<br/>clip-level ECR"]
    O --> P["Mean over clips<br/>final teacher ECR"]
```

### Inputs

```text
Video:
- RGB frames
- audio waveform
- video frames for mPLUG-2

Metadata:
- Title
- Description
```

### Heavy Feature Extractors

```text
EfficientNetV2-s:
  semantic visual feature, 528d/frame

Distortion / UVQ-style network:
  distortion feature, 256d/frame

ResNet3D-18:
  action/motion feature, 512d/clip

mPLUG-2:
  generated caption text
  mid-layer video feature, 1024d/clip

YAMNet:
  top-5 sound classes, converted to text

Stable Diffusion text encoder:
  encodes sound/title/description/caption into 77 x 768 tokens
```

### EVQA Fusion

```text
semantic + distortion -> frame fusion
action feature -> query for text cross-attention
mPLUG feature -> visual-language clip feature
sound/title/description/caption -> text tokens

all clip-level features concat -> 4608d
4608d -> fusion MLP -> 512d
512d clip sequence -> 8 Transformer blocks
Transformer output -> MLP -> ECR per clip
mean clip ECR -> final ECR
```

### Exported Artifacts For KD

When `SNAPUGC_EXPORT_ARTIFACTS=1`, the local wrapper patches a working copy to
save these additional tensors without changing scalar predictions:

```text
- teacher_ecr
- clip_ecr
- fusion_hidden
- temporal_hidden
- caption_feature
- action_feature
- frame_fusion_feature
- text_tokens
- text_pooled
- attention_mean
- attention_importance
```

### DOVER Note

The paper text discusses aesthetic features from DOVER, but the released
`ECR_inference/test_SnapUGC_baseline.py` script and `EVQA.forward` used here do
not load DOVER and do not expose `technical` or `aesthetic` score inputs. The
current run therefore reproduces the official released teacher inference code,
not a custom DOVER-augmented extension.
