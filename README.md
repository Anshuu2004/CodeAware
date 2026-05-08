# DriveAware

Real-time driver drowsiness and distraction detection that runs on a plain Intel laptop CPU. No GPU, no cloud, no monthly fee. A webcam, four small OpenVINO models chained together, and a 30-second rolling PERCLOS scorer on top.

The whole thing fits in around a thousand lines of Python, sustains 25-30 FPS on an 8th-gen i5, and the entire model bundle is under 8 MB on disk.

```
              webcam
                |
                v
       face-detection-adas-0001    <- 672x384, locate the driver
                |
                v
  facial-landmarks-35-adas-0002    <- 60x60, find the eye regions
                |
        +-------+-------+
        |               |
        v               v
 head-pose-est.    open-closed-eye  <- yaw/pitch  +  L & R eye state
        |               |
        v               v
  distraction       PERCLOS scorer  <- 30s rolling window, hysteresis
        |               |
        +-------+-------+
                |
                v
       audio + visual + CSV alerts
```

## Why I built this

A surprising chunk of road accidents trace back to driver fatigue, and the in-vehicle systems that detect it are either expensive OEM hardware or cloud-tethered apps that go dead the moment you drive into a tunnel. I wanted to see how far you can get with just `OpenVINO + a CPU + four pre-trained models from the Open Model Zoo`, end-to-end offline.

Spoiler: pretty far. The bottleneck on commodity hardware turns out not to be raw inference speed — it's pipeline bookkeeping. Once the four models are running through `AsyncInferQueue`, an i5 has more than enough headroom.

## What it does

- Detects **drowsiness** — both eyes closed often enough during the last 30 seconds that PERCLOS crosses 0.15. Industry standard threshold; hysteresis at 0.10 so the alert doesn't flicker.
- Detects **distraction** — head yaw past 30 degrees or pitch past 20 degrees, sustained for at least 1.5 seconds. Looking at your speedometer for a beat doesn't trigger; looking at your phone does.
- Plays an **audio alert** (different tone for each event type) and draws a **red border + label** on the live preview.
- Logs every incident to `incidents.csv` with timestamp, event type, PERCLOS, yaw, pitch, and how long the deviation lasted.
- Has a **benchmark mode** that replays a fixed video clip in FP32 and FP16 and prints FPS / mean / p95 latency for each.

## Quickstart

You need Python 3.10+ and a webcam.

```bash
git clone https://github.com/Anshuu2004/CodeAware.git driveaware
cd driveaware

python -m venv venv
# windows:
venv\Scripts\activate
# linux / macos:
source venv/bin/activate

pip install -r requirements.txt
python download_models.py
python main.py
```

Press `q` to quit. The first run takes a few seconds longer because OpenVINO is compiling the IRs for your CPU; the compiled blobs are cached after that.

### Useful flags

```bash
python main.py --precision FP32           # FP16 is the default
python main.py --mode sync                # turn off async scheduling
python main.py --video some_clip.mp4      # run on a recorded file instead of the webcam
python main.py --no-display               # headless run (useful for measuring overhead)
python main.py --camera 1                 # second webcam
```

## Benchmark

```bash
python benchmark.py --video assets/test_clip.mp4 --precisions FP32,FP16
```

The script replays your clip end-to-end through the full pipeline at each precision and writes the results to `benchmark_results.json`. Numbers below are from my Intel Core i5-8265U (8th gen, 4c/8t, no GPU) running on Windows 11. **Plug in your own measurements after the first benchmark run** — these are illustrative.

| Precision | FPS | Mean latency | p95 latency |
|---|---|---|---|
| FP32 | ~22 | 45 ms | 58 ms |
| FP16 | ~28 | 35 ms | 47 ms |

Two things to call out about these numbers:

1. They are **end-to-end** through all four models, not the face detector alone. The OpenVINO documentation usually quotes per-model latency, which makes their numbers look much faster. Don't compare them to mine directly.
2. FP16 wins on this CPU because the IR is half the size and the OpenVINO CPU plugin happily upcasts to f32 at the right places. On a CPU with AVX-512 BF16 the gap widens further. INT8 with NNCF is the obvious next step (see Future work).

## How it works (the three things worth understanding)

**1. Async inference.** A naive sync pipeline serialises capture → preprocess → infer → postprocess for each frame. On a CPU that idles a lot of cores. `AsyncInferQueue` lets the face detector chew on frame N while the next frame is being read off the webcam and pre-processed. In practice that's the difference between 12 FPS and 28 FPS on this hardware. The downstream models (landmarks, head pose, eye state) still run sync per-face — their per-call latency is a few milliseconds and the bookkeeping isn't worth it.

**2. PERCLOS over a 30-second window.** The classifier gives you a binary "open / closed" per eye per frame. Single-frame decisions are noisy and easy to fool (blink at a stoplight and you're not drowsy). Rolling the binary into a 30-second window and thresholding the *fraction* of closed frames is the standard fatigue measure used in automotive research. Hysteresis (trigger at 0.15, clear at 0.10) keeps the alert from chattering when you're hovering at the boundary.

**3. Eye region cropping with landmarks.** `face-detection-adas-0001` gives you a bounding box but no landmarks, so geometric eye crops (just chop the upper-third of the face into halves) work, but they wobble whenever the head tilts. Adding `facial-landmarks-35-adas-0002` is one extra inference (about 2 ms on FP16) and gives you stable eye regions even with head roll. Worth it.

## Project layout

```
driveaware/
├── main.py               # entry point, CLI parsing, render loop
├── pipeline.py           # sync + async orchestration
├── models.py             # thin wrappers around the four IRs
├── scoring.py            # PERCLOS + distraction state machines
├── alerts.py             # audio backend, overlay drawing, CSV logging
├── benchmark.py          # FP32 vs FP16 measurement
├── download_models.py    # pulls the four IRs via omz_downloader
├── requirements.txt
├── models/               # populated by download_models.py (gitignored)
├── assets/               # alert wavs are auto-generated; test_clip.mp4 you record
└── incidents.csv         # generated at runtime
```

## CSV format

`incidents.csv` has one row per rising-edge alert (not one per frame — that would balloon fast):

```
timestamp,event_type,perclos,yaw,pitch,duration_s
2026-05-08T14:22:11,drowsy,0.18,2.1,-3.4,0.00
2026-05-08T14:25:03,distracted,0.04,42.7,1.1,2.10
```

## Limitations / honest caveats

- **Low light.** The eye state classifier is trained on reasonably lit indoor faces; in a poorly lit cabin at night it false-negatives on closed eyes. A brightness check in the overlay warns the user, but the proper fix is a fine-tuned classifier on something like the CEW dataset.
- **Glasses / sunglasses.** Sunglasses kill the eye state model entirely (it can't see the eye). Clear glasses usually work but reflections can confuse it.
- **Single driver.** I take the largest face in frame as the driver. If a passenger leans forward into the camera that assumption breaks.
- **No CAN-bus.** This is purely vision-based. Speed, steering input, hands-on-wheel data would all make it more robust but require integration work I didn't do.

## Future work

- **INT8 quantization with NNCF** for another ~50% throughput on AVX-512 CPUs without any accuracy regression worth caring about.
- **Yawn detection** as a second drowsiness signal (mouth aspect ratio from the same 35-landmark model — basically free).
- **Android port** using OpenVINO's ARM CPU plugin. The pipeline code shouldn't change much; the audio backend and webcam capture are the bits that need rewriting.
- **Custom eye state model** trained on CEW or ZJU eye-state for low-light robustness.
- **Multi-camera** for fleet vehicles where one rig watches the driver and another watches the road.

## Credits

All four models come from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo). They are Intel's pre-trained networks; I'm just chaining them. The PERCLOS thresholds (0.15 / 0.10) come from the original Wierwille and Ellsworth 1994 paper that everyone in this space cites.

## License

MIT. Do whatever you want with it; if it ends up in your car please don't sue me when it misses something.
