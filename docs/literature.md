# Literature Alignment

This repo keeps only the final thesis pipeline: SnapUGC-LightKD.

## Main References

- Delving Deep into Engagement Prediction of Short Videos, ECCV 2024: https://arxiv.org/abs/2410.00289
- VQualA 2025 Challenge on Engagement Prediction for Short Videos: https://arxiv.org/abs/2509.02969
- Engagement Prediction of Short Videos with Large Multimodal Models, ICCVW 2025: https://openaccess.thecvf.com/content/ICCV2025W/VQualA/html/Sun_Engagement_Prediction_of_Short_Videos_with_Large_Multimodal_Models_ICCVW_2025_paper.html
- MMF-QE, ICCVW 2025: https://openaccess.thecvf.com/content/ICCV2025W/VQualA/html/Guan_MMF-QE_Advanced_Multi-Modal_Fusion_for_Quality_Assessment_and_Engagement_Prediction_ICCVW_2025_paper.html

## Model Choices

- CLIP ViT-B/16: visual semantic frame representation.
- R(2+1)D-18 Kinetics-400: lightweight motion/action representation.
- DOVER: video UGC technical/aesthetic quality, replacing image-only MUSIQ/TOPIQ.
- YAMNet: lightweight AudioSet background sound representation.
- Sentence-T5-base: T5-family sentence embeddings for title/description/caption/rationale.
- Qwen2.5-VL-3B-Instruct: offline caption/rationale annotator, not used as deployed student input.

## KD Design

Teacher receives privileged multimodal signals: visual, motion, quality, audio, metadata, caption, and rationale.
Student receives lightweight deployable signals: visual, audio, and metadata.

The KD objective combines ground-truth ECR, teacher soft ECR, hidden representation distillation, temporal attention distillation, and auxiliary aesthetic/technical supervision.
