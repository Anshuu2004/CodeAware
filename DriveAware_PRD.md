# DriveAware — Product Requirements Document

**Real-Time Driver Drowsiness & Distraction Detection on Intel CPUs**

| Field | Value |
|---|---|
| Author | Abhishek Choudhary |
| Status | In development |
| Target completion | 3 days from kickoff |
| Repo | `github.com/Anshuu2004/driveaware` |
| Demo target | Live webcam, Intel Core i5 laptop, no GPU |

---

## 1. Problem & Motivation

Driver fatigue causes ~20% of road accidents globally. Existing in-vehicle solutions are either expensive OEM systems (Bosch, Aptiv) or cloud-dependent apps that fail in tunnels/rural areas. **DriveAware is a fully offline, edge-deployed perception system that runs at 30 FPS on a stock laptop CPU** — no GPU, no internet, no monthly fees.

The technical bet: chain three small pre-trained OpenVINO Model Zoo models through the Async Inference API, layer a PERCLOS scoring window on top, and prove that real-time driver monitoring works on commodity Intel hardware. This is the kind of project Intel's edge-AI team actually cares about — small models, big throughput, measurable latency wins.

## 2. Goals & Non-Goals

### Goals (must hit for v1)
- Detect drowsiness (eye-closure events) in real time from a webcam feed
- Detect distraction (sustained head-pose deviation) in real time
- Run at ≥25 FPS sustained on an Intel Core i5 laptop CPU
- End-to-end latency under 80 ms (capture → alert)
- Audible alert + on-screen warning when drowsy/distracted
- CSV log of incidents with timestamps for post-drive review
- Side-by-side FP32 vs FP16 IR benchmark numbers (proves the optimization claim on the resume)

### Non-Goals (explicitly out of scope for v1)
- Mobile/Android port
- Multiple driver profiles
- Cloud sync / remote dashboard
- Custom-trained models (we use Model Zoo verbatim)
- INT8 quantization with NNCF (nice-to-have for v2)
- Heart-rate, yawn detection, hands-on-wheel detection
- Vehicle CAN-bus integration

## 3. Users & Use Cases

| User | Primary Use Case |
|---|---|
| Long-haul drivers | Background safety net during night drives |
| Fleet operators | Per-vehicle incident logs to identify high-risk routes/drivers |
| Driving schools | Live feedback during student practice sessions |
| Demo audience (Intel) | Visible proof that Intel CPU + OpenVINO can do real-time CV |

The last row is the only one that actually matters in the next 3 days. Build for the demo.

## 4. Functional Requirements

### F1 — Webcam Capture
- Grab frames from default webcam at 640×480 @ 30 FPS via OpenCV
- Graceful fallback if no webcam (log error, exit cleanly)

### F2 — Face Detection
- Use `face-detection-adas-0001` from OpenVINO Model Zoo
- Return the largest face bounding box per frame (driver = closest face)
- If no face for >2 seconds, show "Driver not visible" warning

### F3 — Head Pose Estimation
- Use `head-pose-estimation-adas-0001`
- Output yaw, pitch, roll for the detected face
- Distraction = |yaw| > 30° OR |pitch| > 20° sustained for >1.5 seconds

### F4 — Eye State Classification
- Use `open-closed-eye-0001`
- Need to crop eye regions first — `face-detection-adas-0001` does NOT give landmarks, so we'll use `facial-landmarks-35-adas-0002` to locate eyes precisely. Adds 1 inference per frame but is reliable.
- Classify each eye independently: open (0) or closed (1)

### F5 — PERCLOS Drowsiness Scoring
- Maintain a rolling 30-second window of (timestamp, both_eyes_closed) tuples
- PERCLOS = fraction of window where both eyes were closed
- Drowsy threshold: PERCLOS > 0.15 (industry standard)
- Hysteresis: trigger alert at 0.15, clear at 0.10 (prevents flicker)

### F6 — Async Inference Pipeline
- Use OpenVINO's `AsyncInferQueue` so frame N+1's preprocessing overlaps frame N's inference
- Target: face detection + landmarks + head pose + 2× eye state per frame, fully overlapped

### F7 — Alerts
- Audible: `simpleaudio` or `playsound`, two distinct tones (drowsy = lower, distracted = higher)
- Visual: red overlay border + text label on the live preview window
- Cooldown: don't replay alert sound for 5 seconds after the last one

### F8 — Incident Logging
- CSV at `incidents.csv` with columns: `timestamp, event_type, perclos, yaw, pitch, duration_s`
- Append-only, flushed every event

### F9 — FP32 vs FP16 Benchmark
- CLI flag `--precision fp32|fp16`
- Print FPS, mean latency, p95 latency on exit
- Reproducible: fixed 60-second test using a recorded video file (`test_clip.mp4`) so numbers don't depend on lighting

## 5. Non-Functional Requirements

| Aspect | Requirement |
|---|---|
| Latency | <80 ms capture-to-alert end-to-end |
| Throughput | ≥25 FPS sustained on Intel Core i5 (8th gen+) |
| Memory | <500 MB RSS |
| Cold start | <5 seconds (model load + first inference) |
| Footprint | Single-digit MB of code; models downloaded on first run |
| OS | Windows 10/11 + Ubuntu 22.04 (test on whichever you actually have) |

## 6. Architecture

```
Webcam ──► Frame Buffer ──► AsyncInferQueue
                                  │
                                  ├─► face-detection-adas-0001
                                  │         │
                                  │         ▼
                                  │   ┌──────────────────────────┐
                                  │   │ For each detected face:  │
                                  │   │  ├─ landmarks-35         │
                                  │   │  ├─ head-pose-estimation │
                                  │   │  ├─ open-closed-eye (L)  │
                                  │   │  └─ open-closed-eye (R)  │
                                  │   └──────────────────────────┘
                                  ▼
                       ┌────────────────────────┐
                       │  PERCLOS Scorer (30s)  │
                       │  Distraction Detector  │
                       └────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            Audio Alerts    Visual Overlay   CSV Logger
```

### File layout
```
driveaware/
├── README.md
├── PRD.md                       # this doc
├── requirements.txt
├── main.py                      # entry point, CLI flag parsing
├── pipeline.py                  # AsyncInferQueue orchestration
├── models.py                    # model loading + preprocessing
├── scoring.py                   # PERCLOS window + distraction logic
├── alerts.py                    # audio + visual + CSV
├── benchmark.py                 # FP32 vs FP16 measurement script
├── models/                      # downloaded IRs (.gitignored except README)
│   └── README.md                # how to fetch with omz_downloader
├── assets/
│   ├── alert_drowsy.wav
│   ├── alert_distracted.wav
│   └── test_clip.mp4            # 60s benchmark video
└── incidents.csv                # generated at runtime
```

## 7. Models — Exact Names & Sources

All from `Open Model Zoo`. Fetch with:
```bash
omz_downloader --name face-detection-adas-0001
omz_downloader --name facial-landmarks-35-adas-0002
omz_downloader --name head-pose-estimation-adas-0001
omz_downloader --name open-closed-eye-0001
```

| Model | Input | Output | Approx Size (FP16) |
|---|---|---|---|
| face-detection-adas-0001 | 672×384 BGR | bboxes [n, 7] | ~1 MB |
| facial-landmarks-35-adas-0002 | 60×60 BGR | 70 floats (35 xy) | ~2 MB |
| head-pose-estimation-adas-0001 | 60×60 BGR | yaw, pitch, roll | ~3.5 MB |
| open-closed-eye-0001 | 32×32 BGR | softmax [open, closed] | <100 KB |

Total disk: <8 MB. Total RAM during inference: well under 200 MB.

## 8. Implementation Plan — 3 Days

### Day 1 — Skeleton + Synchronous Pipeline (4-5 hours)

**Hour 1: Setup**
```bash
python -m venv venv
# Activate: source venv/bin/activate  (Linux/Mac) or venv\Scripts\activate  (Windows)
pip install openvino openvino-dev opencv-python numpy simpleaudio
mkdir models && cd models
omz_downloader --name face-detection-adas-0001 --output_dir .
omz_downloader --name facial-landmarks-35-adas-0002 --output_dir .
omz_downloader --name head-pose-estimation-adas-0001 --output_dir .
omz_downloader --name open-closed-eye-0001 --output_dir .
cd ..
```

**Hours 2-3: Sync inference loop (correctness first, optimization later)**
- `models.py`: write `load_model(path, device='CPU')` returning a `compiled_model`
- `main.py`: webcam loop → preprocess frame → run face detection → draw bbox → show
- Confirm: face box follows your face, terminal shows raw FPS

**Hours 4-5: Add the other three models, still synchronous**
- Chain landmarks → head pose → eye state crops (use landmark indices for eye corners)
- Print yaw/pitch/roll and L/R eye states to terminal
- Don't worry about FPS yet — it'll be ugly (~10-12 FPS sync). That's fine.

**End-of-day deliverable:** Webcam window with face bbox + on-screen text showing yaw/pitch + "EYES: OPEN/CLOSED". Commit with tag `v0.1-sync-pipeline`.

### Day 2 — Async Pipeline + Scoring Logic (5-6 hours)

**Hours 1-2: Convert to AsyncInferQueue**
- This is the meaty part. Read the official OpenVINO notebook `407-person-tracking-webcam` for the exact pattern.
- Pattern: enqueue inference for frame N+1 immediately after preprocessing; pick up frame N's results in the callback.
- Sanity check: FPS should jump from ~10-12 to 25-30+. If it doesn't, you're still serializing somewhere.

**Hours 3-4: PERCLOS scorer + distraction detector**

`scoring.py`:
```python
from collections import deque
import time

class PerclosScorer:
    def __init__(self, window_s=30, threshold=0.15, clear_threshold=0.10):
        self.window_s = window_s
        self.events = deque()
        self.threshold = threshold
        self.clear_threshold = clear_threshold
        self.is_drowsy = False

    def update(self, both_closed: bool) -> bool:
        now = time.time()
        self.events.append((now, both_closed))
        while self.events and now - self.events[0][0] > self.window_s:
            self.events.popleft()
        if not self.events:
            return False
        score = sum(c for _, c in self.events) / len(self.events)
        if score > self.threshold:
            self.is_drowsy = True
        elif score < self.clear_threshold:
            self.is_drowsy = False
        return self.is_drowsy


class DistractionDetector:
    def __init__(self, yaw_thresh=30.0, pitch_thresh=20.0, sustain_s=1.5):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.sustain_s = sustain_s
        self.deviation_start = None

    def update(self, yaw: float, pitch: float) -> bool:
        deviating = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh
        now = time.time()
        if deviating:
            if self.deviation_start is None:
                self.deviation_start = now
            return (now - self.deviation_start) > self.sustain_s
        else:
            self.deviation_start = None
            return False
```

**Hours 5-6: Visual + audio alerts + CSV logging**
- Red border overlay when drowsy or distracted
- Text label: "DROWSY!" or "DISTRACTED!"
- Audio: play wav with cooldown (5s)
- CSV: append on alert state-change (rising edge only), not every frame

**End-of-day deliverable:** Close your eyes for 4-5 seconds → alert fires. Look hard left for 2 seconds → distraction alert fires. CSV grows. Commit `v0.2-full-pipeline`.

### Day 3 — Benchmarking, Polish, Demo Recording (3-4 hours)

**Hour 1: FP32 vs FP16 benchmark**
- `benchmark.py`: replay `test_clip.mp4` (record 60 seconds of yourself driving-simulated), run both precisions
- Output JSON: `{"fp32": {"fps": ..., "mean_ms": ..., "p95_ms": ...}, "fp16": {...}}`
- This produces the numbers that go on the resume. Use the *real* numbers — don't overclaim.

**Hour 2: README**

Sections to include:
- One-line description, GIF/screenshot of it working
- Architecture diagram (paste from PRD)
- Quickstart: `pip install -r requirements.txt && python download_models.py && python main.py`
- Benchmark results table (FP32 vs FP16)
- "How it works" — 3 paragraphs explaining async inference, PERCLOS, head-pose thresholds

**Hour 3: Demo video**
- 30-second screen recording (OBS or Windows Game Bar): show normal driving → close eyes → drowsy alert → look away → distraction alert → CSV preview
- Upload to YouTube unlisted, link from README

**Hour 4: Buffer / fix the thing that's broken**
There's always one. Use this hour for it. If nothing's broken, push to GitHub and you're done early.

**End-of-day deliverable:** Pushable, demoable, defensible project. Commit `v1.0`.

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Async pattern is harder than expected, eats Day 2 | High | Medium | Fallback: ship synchronous v1, mention "async optimization in progress" honestly. Sync at 12-15 FPS is still demoable. |
| Eye-state model misclassifies in low light | Medium | Medium | Add a brightness check; warn user if frame too dark. Don't try to fix the model. |
| Webcam crop produces bad eye regions | Medium | High | Use the 35-landmark model (built into the plan) instead of geometric eye crops. |
| Bhavya Ma'am asks for INT8 quantization on the call | Low | Low | Honest answer: "FP16 was sufficient for real-time on CPU; INT8 with NNCF is the natural next step and I have it on the v2 list." This is true and reasonable. |
| Laptop is too old, can't hit 25 FPS even async | Low | High | Drop input resolution to 320×240 for face detection (it's still accurate); drop to 15 FPS target and update resume claims to match what you actually measure. |

## 10. What Goes on the Resume After Build

The resume currently says ~30 FPS at sub-60ms latency. **Measure first, edit the resume to match reality second.** If you hit 28 FPS at 65ms, write 28 FPS at 65ms — the precision is *more* impressive than a round number anyway, because it shows you actually measured.

If you fall short of 25 FPS, change the resume to "real-time (sustained 15 FPS)" and explain in the interview that head-pose was the bottleneck. That's still a defensible result.

## 11. Out-of-Scope / Future Work (good to mention if asked)

- INT8 quantization with NNCF post-training quantization (target: 50%+ additional throughput on AVX-512 CPUs)
- Replace `open-closed-eye-0001` with a fine-tuned model on the CEW or ZJU eye-state dataset for low-light robustness
- Android port using OpenVINO's ARM CPU plugin
- Yawn detection (mouth aspect ratio) as a third drowsiness signal
- Multi-camera support for fleet vehicles

## 12. Definition of Done

- [ ] `python main.py` runs with no errors on a fresh clone after `pip install -r requirements.txt`
- [ ] FPS counter visible on screen during live operation
- [ ] Drowsiness alert demonstrably fires within 5 seconds of closing eyes
- [ ] Distraction alert demonstrably fires within 2 seconds of head turn
- [ ] `incidents.csv` contains entries from the demo run
- [ ] `benchmark.py` outputs FP32 vs FP16 comparison
- [ ] README has architecture diagram, quickstart, and benchmark results
- [ ] 30-second demo video linked from README
- [ ] Repo pushed to `github.com/Anshuu2004/driveaware`, public
- [ ] Resume bullets updated to match *measured* numbers
