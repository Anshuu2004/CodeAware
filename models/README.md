# Models

This folder is where the four Open Model Zoo IRs land after you run
`python download_models.py` from the project root. The downloads
themselves are gitignored — only this readme is checked in.

What goes here, after a successful download (FP16):

```
models/
└── intel/
    ├── face-detection-adas-0001/FP16/face-detection-adas-0001.{xml,bin}
    ├── facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.{xml,bin}
    ├── head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.{xml,bin}
    └── open-closed-eye-0001/FP16/open-closed-eye-0001.{xml,bin}
```

If you want to run the FP32-vs-FP16 benchmark, grab both:

```bash
python download_models.py --precisions FP16,FP32
```

## Manual download (if `omz_downloader` misbehaves)

```bash
omz_downloader --name face-detection-adas-0001         --output_dir models
omz_downloader --name facial-landmarks-35-adas-0002    --output_dir models
omz_downloader --name head-pose-estimation-adas-0001   --output_dir models
omz_downloader --name open-closed-eye-0001             --output_dir models
```

`omz_downloader` ships with `openvino-dev`, which is in `requirements.txt`.

## Model summary

| Model | Input | Output | Purpose |
|---|---|---|---|
| face-detection-adas-0001 | 672x384 BGR | bboxes [N, 7] | locate the driver's face |
| facial-landmarks-35-adas-0002 | 60x60 BGR | 35 (x, y) points | find the eye regions |
| head-pose-estimation-adas-0001 | 60x60 BGR | yaw / pitch / roll | distraction signal |
| open-closed-eye-0001 | 32x32 BGR | softmax(open, closed) | drowsiness signal |

Total disk footprint after FP16 download: under 8 MB.
