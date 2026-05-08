# Assets

Three things live (or should live) here:

- `alert_drowsy.wav` — generated automatically on first run by `alerts.py`. A 440 Hz tone, 0.6 s long. Lower / mellow.
- `alert_distracted.wav` — generated automatically on first run by `alerts.py`. An 880 Hz tone, 0.6 s long. Higher / urgent.
- `test_clip.mp4` — record this yourself. ~60 seconds of you sitting at a desk, looking around, occasionally closing your eyes for a few seconds. The benchmark script replays this so the FP32-vs-FP16 numbers are reproducible across runs.

The wavs and the mp4 are all gitignored. The wavs are tiny and regenerated on demand; the mp4 is something you record once and keep locally.

## Recording the test clip

Anything works — phone camera, laptop webcam, OBS. Save the file as `assets/test_clip.mp4`. Aim for:

- ~60 seconds total
- Mostly looking ahead, with a few deliberate "look left" / "look right" / "look down" moments past the distraction thresholds
- A couple of 4-5 second eye-closure stretches so the PERCLOS scorer fires
- Reasonable lighting (the eye classifier struggles in dim/backlit scenes)

Then:

```bash
python benchmark.py --video assets/test_clip.mp4 --precisions FP32,FP16
```
