
AR + Mouse + AI Shapes (Gesture Control)

Touchless Gesture Mouse + 3D Object AR Manipulation + Air Drawing
Computer Vision Project using Python, OpenCV, MediaPipe, Kalman Filter
========================================

📍 Project Description
----------------------
This software allows users to interact with the computer
using their HAND GESTURES — without touching the mouse.

Main features:
✔ Control mouse pointer using touchless gestures
✔ Click left/right using pinch gestures
✔ 3D AR Object control (rotate, scale, move)
✔ Air Drawing Mode (sketch in air with webcam)
✔ AI Shape Generation using voice commands (optional)

----------------------------------------
🛠️ Technologies Used
----------------------------------------
• Python 3.9+
• OpenCV (cv2)
• MediaPipe Hands
• NumPy
• PyAutoGUI
• Pynput (Mouse automation)
• Kalman Filter (cursor smoothing)
• Google Gemini API (optional)
• SpeechRecognition API (optional)

----------------------------------------
🎮 Modes & Gesture Controls
----------------------------------------
Mode Switch:
• Key 1 → 3D Shape Mode
• Key 2 → AI Shape Mode
• Key 3 → Mouse Mode
• Key 4 → Air Drawing Mode
• Key Q → Quit Program
• Key A → Auto-Rotate ON/OFF

3D Shape AR Controls:
• Two hands open → Scale + Rotate object
• Two index fingers → Move object
• One index finger → Move object (slow)

Mouse Gesture Controls:
• Point index finger → Move Cursor
• Pinch (index + thumb) → Left Click
• Pinch (index + middle + thumb) → Right Click
• Closed fist → Drag & Hold

Air Drawing Controls:
• Index finger UP → Draw
• Fist → Stop drawing
• Top buttons → Switch colors (Blue, Green, Red, Yellow)
• CLEAR button → Clear canvas

----------------------------------------
🎙️ Optional Voice + AI Features (Disabled by default)
----------------------------------------
You can say:
• "Letter A"
• "Number 5"
• "Triangle"
• "Pentagon"
• "Generate shape: star" (AI generated)

To enable these:
See Setup Instructions below.

----------------------------------------
📦 Installation Instructions
----------------------------------------
Run these commands:

pip install opencv-python mediapipe numpy pynput pyautogui
pip install google-generativeai speechrecognition pyaudio

(If PyAudio installation fails, follow OS-specific guide)

----------------------------------------
📷 Hardware Requirements
----------------------------------------
• A working Webcam
• Computer with decent CPU for real-time tracking

----------------------------------------
🔐 Security & Code Protection
----------------------------------------
For security reasons:
• Gemini API Key is NOT included
• Voice commands are disabled publicly

If you have your own Gemini Key:
Search this in code:
"GEMINI_API_KEY_HERE"

Replace it with your API key:
GENAI_API_KEY = "YOUR_API_KEY"

Then remove this line:
GEMINI_AVAILABLE = False

Similarly to enable voice:
SR_AVAILABLE = True

----------------------------------------
📌 File Usage
----------------------------------------
Run program using:

python main.py

Default window name:
"AR + Mouse + AI Shapes"

Press Esc or Q to close safely.

----------------------------------------
📌 Known Limitations
----------------------------------------
• Better performance in bright lighting
• Not optimized for older webcams
• Voice recognition requires a clear microphone

----------------------------------------
👤 Author
----------------------------------------
Created by: Harshit Shaw
Project: Gesture-Controlled AR Interface System
Version: Public Release v1.0

----------------------------------------
📄 License
----------------------------------------
This project is for EDUCATIONAL use only.
Copying or submitting this as your own may be prohibited.
Credit to original author required.

========================================
THANK YOU FOR USING THIS SOFTWARE 😊
========================================
