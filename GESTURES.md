# Real-Time Gesture Control

This project implements real-time, controller-free gesture interaction using a webcam, OpenCV and MediaPipe.

## Supported Gestures & Mappings

- Wave (left/right / up/down)
  - Rotates the hologram horizontally or tilts it vertically.
- Pinch (thumb + index)
  - Single pinch: left click.
  - Pinch + middle (or multi-finger): right click.
- Two-Hand Pinch / Spread
  - Two hands open and moved apart/together: scale the 3D object (zoom).
- Two Index Fingers
  - Move the object (translation in XY plane).
- Swipe (left/right/up/down)
  - (Planned) navigation between models or UI show/hide—basic swipe support can be extended.
- Palm Push / Pull
  - Push forward / pull back to adjust object depth (Z axis) — depth estimation approximated by hand center distances.
- Grab (fist) + Hold
  - Grab & hold to drag object; release to drop.
- Two-Hand Rotate
  - Use both hands to rotate object 360°.

## How it Works (high level)

- Hand landmarks are detected with MediaPipe Hand Landmarker (Tasks API).
- A small Kalman filter smooths landmark positions for stable cursor/object control.
- Gestures are inferred from landmark geometry (tip vs pip/dip positions, distances between points).
- The app maps gestures to transformations on the 3D wireframe renderer in `hand.py`.

## Try it locally

1. Install dependencies from `requirements.txt` or manually:

```powershell
pip install -r requirements.txt
```

2. Run the demo (webcam required):

```powershell
python hand.py
```

3. Use the keyboard shortcuts to switch modes:

- `1` — 3D Shape Mode (AR)
- `2` — AI Voice Shape Mode (if configured)
- `3` — Gesture Mouse Mode
- `4` — Air Drawing Mode
- `A` — Toggle auto-rotate
- `Q` — Quit

## Notes for Contributors

- The gesture logic and mapping live inside `hand.py` (functions like `handle_ar_gestures`, `detect_mouse_gesture`, and `handle_air_draw_mode`).
- Improvements that would complete issue #5 further:
  - Add configurable thresholds for different camera setups.
  - Add explicit swipe-navigation handlers and UI feedback for selection.
  - Add unit tests for gesture inference logic (pure-Python functions isolated from camera I/O).

If you'd like, I can open a PR with these docs and a small CLI branch to the upstream repo — tell me whether you want me to create the branch and the exact PR title/body.
