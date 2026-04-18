# Data Layout

This directory is intentionally git-ignored except this README.

Expected local layout:

```text
data/
  train_data.csv
  val_data.csv
  train_videos/
    <video_id>.mp4
```

In this local workspace, these paths are symlinks to the original SnapUGC data from the old working repo to avoid duplicating multi-GB videos.
