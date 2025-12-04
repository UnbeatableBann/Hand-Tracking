# Hand–Boundary Interaction POC

Real-time hand tracking + virtual danger boundary (no MediaPipe / OpenPose)

## Objective

Prototype a system that:

* Tracks the user’s hand in **real time** using classical computer vision (no pose APIs).
* Draws a **virtual object/boundary** on screen.
* Detects when the hand approaches or touches this boundary.
* Displays a clear warning:

  * `SAFE`
  * `WARNING`
  * `DANGER` (with the on-screen text: **DANGER DANGER**)

Target performance: **≥8 FPS** on CPU.

---

## Features

### 1. Real-Time Hand Detection (Classical CV)

No MediaPipe, no OpenPose.
Uses:

* Gaussian blur
* HSV skin-color segmentation
* Morphological filtering
* Contour extraction & convex hull
* Simple fingertip estimation (topmost contour point)

### 2. Virtual Boundary

A rectangle drawn in the center of the frame.
Distance from hand → rectangle controls the system state.

### 3. Distance-Based State Logic

* **SAFE**: hand far from boundary
* **WARNING**: hand approaching
* **DANGER**: hand extremely close or touching
  Shows **“DANGER DANGER”** overlay in red.

### 4. Visual Feedback

Camera stream overlays:

* Current state
* Fingertip location
* Minimum distance to boundary
* Hand contour + convex hull
* FPS
* Virtual rectangle

### 5. Performance

* CPU-only
* Works at 8–25 FPS depending on camera and resolution
* No deep pose models needed

---

## Demo Screenshot (Concept)

```
+--------------------------------------------------------+
|  STATE: WARNING                                        |
|                                                        |
|                [ Virtual Rectangle ]                   |
|                      DANGER ZONE                       |
|                                                        |
|        Hand Contour + Fingertip → o                    |
|                                                        |
+--------------------------------------------------------+
```

---

## Requirements

### Python Dependencies

```
opencv-python
numpy
```

---

## How to Run

1. Install dependencies:

```bash
pip install opencv-python numpy
```

2. Run the script:

```bash
python hand_boundary_poc.py
```

3. Press **ESC** to exit.

4. Optional tuning (during runtime):

* `+` → reduce SAFE/WARN distance
* `-` → increase SAFE/WARN distance

---

## File: `hand_boundary_poc.py`

This file contains:

* Camera capture
* Skin segmentation
* Contour extraction
* Fingertip detection
* Virtual rectangle
* Distance logic
* State rendering
* FPS counter

You already have the full code in the previous output.

---

## How It Works (Short Version)

1. Convert frame → HSV
2. Threshold by skin color → binary mask
3. Clean mask using morphological operations
4. Extract largest contour → assume hand
5. Fingertip = topmost contour point
6. Compute shortest distance from hand contour → virtual rectangle
7. Map distance to SAFE / WARNING / DANGER
8. Draw overlays onto frame

---

## Tuning

### If detection is unstable:

* Adjust HSV skin ranges inside the script:

```python
SKIN_HSV_LOWER = np.array([0, 30, 60])
SKIN_HSV_UPPER = np.array([25, 255, 255])
```

### If FPS is low:

* Reduce `FRAME_WIDTH` to `320`
* Disable extra debug windows
* Reduce morphological iterations

### If fingertips are wrong:

Replace fingertip logic with convexity defects or curvature points.
I can generate that code if needed.

---

## Limitations

* Skin segmentation depends on lighting.
* Fingertip method assumes finger is pointed upward.
* Complex backgrounds may cause noise.
* Single-hand assumption.
"# Hand-Tracking" 
