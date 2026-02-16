
<h1 align="center">✋ AR + Mouse + AI Shapes (Gesture Control System)</h1>

<p align="center">
A real-time AI + Hand Gesture based Human-Computer Interaction system using OpenCV and MediaPipe.
</p>

---

## 🚀 Project Overview

This project enables **touchless interaction** with a computer using only your **hand gestures** via a webcam.

With different modes, users can:
- Manipulate and interact with **3D AR shapes**
- Control mouse **pointer + clicks + drag**
- Draw in the air on screen
- Use **voice commands** for AI-generated shapes *(optional)*

Ideal for:
> Computer Vision | AR/VR | Human-Computer Interaction | Gesture UI Projects

---
## 📌 Tracking Technology

This system uses **Kalman Filter based motion smoothing** to ensure
high-stability cursor control and AR object interaction.

Why Kalman Filter?
- Reduces jitter from hand detection noise
- Predicts next cursor state for smoother movement
- Makes AR gestures feel natural and responsive

Used for:
✔ Mouse cursor tracking  
✔ AR shape scaling & rotation  
✔ Air drawing stabilization  

## 🧠 Features

| Feature | Description |
|--------|-------------|
| 🧊 AR Object Mode | Rotate, scale, move 3D shapes using gesture |
| 🎙️ AI Shape Mode *(optional)* | Convert voice commands into custom shapes |
| 🖱️ Touchless Mouse Mode | Move cursor, left/right click, drag |
| ✍️ Air Drawing Mode | Draw using index finger with multiple colors |

---

## 🎮 Controls

### Keyboard
| Key | Action |
|-----|--------|
| `1` | 3D Shape Mode |
| `2` | AI Voice Shape Mode |
| `3` | Gesture Mouse Mode |
| `4` | Air Drawing Mode |
| `A` | Auto-Rotate Toggle |
| `Q` | Quit |

### Gesture Mapping
| Gesture | Result |
|--------|--------|
| 1 finger up | Move cursor / draw |
| Pinch (index + thumb) | Left Click |
| Pinch (index + middle + thumb) | Right Click |
| Closed Fist | Drag Mode |
| Two hands open | Scale + Rotate AR object |

---

## 🛠 Installation

```sh
pip install opencv-python mediapipe numpy pynput pyautogui
pip install google-generativeai speechrecognition pyaudio
```

---

## ▶️ Run Application

```sh
python main.py
```

---

## 📌 Requirements
- Python 3.x
- Working webcam
- Microphone *(only if enabling voice features)*

---

## 🔐 AI + Voice Control (Security Disabled in Public)

This GitHub version **does not contain**:
✔ API Keys  
✔ Active voice processing  

To enable:
1. Insert your Gemini API Key in code
2. Change to:
```py
GEMINI_AVAILABLE = True
SR_AVAILABLE = True
```
3. Uncomment voice listener thread in `main()`

Voice Commands Examples:
> "Letter A", "Number 7", "Cube", "Pentagon", etc.

---
---

## 🔧 Performance Tips

- Use bright lighting for better hand detection  
- Keep your hand in frame  
- Plain background improves tracking  

---

## 👤 Author

| Name | Role | Version |
|------|------|---------|
| Harshit Shaw | Developer / Creator | v1.0 Public Release |
| Yash Koparde | Contributor | v1.0 Open Source Contribution |


---

## 📄 License

This project is licensed for **EDUCATIONAL USE ONLY**.  
Redistribution without credit is prohibited.  
