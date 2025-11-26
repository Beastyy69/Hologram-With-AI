AIR MOUSE + AR 3D SHAPES + HAND GESTURE DRAWING + AI SHAPE GENERATION
====================================================================

This project is a complete **Hand Gesture Controlled Interaction System**
using **Computer Vision (OpenCV + MediaPipe)** that supports:

✔ Air Mouse (Control mouse cursor using hand gestures)  
✔ AR 3D Shapes (Manipulate objects using gestures)  
✔ AI Generated Shapes (via Gemini Voice Commands)  
✔ Air Drawing Mode (Draw in the air using finger gestures)

The system uses **hand tracking**, gesture recognition, and voice commands
to switch between different interactive modes in real-time.

--------------------------------------------------------------------
✨ FEATURES
--------------------------------------------------------------------
1️⃣ Mouse Mode (Trackpad Simulation)
-----------------------------------
• Move cursor by index finger inside trackpad boundary  
• Fist gesture → Click & Drag  
• Thumb + Index pinch → Left click  
• Thumb + Index + Middle pinch → Right click  

2️⃣ AR 3D Shape Mode
-------------------
• Rotate object → Move both hands sideways/up-down (palms open)  
• Scale object → Move both hands apart or closer  
• Move object → Two index fingers pointing  
• Auto-rotation toggle using key “A”  
• Built-in shapes: Cube, Pyramid, Sphere, Pentagon, Hexagon, Octagon, Rhombus

3️⃣ AI Shape Generation (Voice Commands)
---------------------------------------
• Create Letters:  
  "letter A", "alphabet C", "A", "B", etc.

• Create Numbers:  
  "number five", "digit 7", "5", "two", etc.

• Custom shapes using Gemini Model

4️⃣ Air Drawing Mode ✏️
-----------------------
• Use index finger to draw  
• Toolbar supports:
  → Blue, Green, Red, Yellow color selection  
  → Clear canvas button  

--------------------------------------------------------------------
🧠 VOICE COMMANDS
--------------------------------------------------------------------
Switch to **AI Mode** to enable voice control:

Letters:
- “letter A”, “B”, “alphabet C”

Numbers:
- “digit 7”, “number three”, “5”

Shapes:
- “cube”, “pyramid”, “sphere”, “pentagon”, “triangle”, etc.

Custom:
- Describe any shape you want (AI will model it)

--------------------------------------------------------------------
🎮 UI CONTROLS (Keyboard)
--------------------------------------------------------------------
1 → AR Built-in Shape Mode  
2 → AI Shape Mode  
3 → Air Mouse Mode  
4 → Air Draw Mode  
A → Toggle auto rotate  
Q → Quit  

--------------------------------------------------------------------
🛠 REQUIREMENTS
--------------------------------------------------------------------
Python 3.8+ recommended

Libraries:
- opencv-python
- mediapipe
- numpy
- pynput
- pyautogui
- SpeechRecognition
- google-generativeai (optional if using AI mode)
- pyaudio (for microphone input)

Install all dependencies:
> pip install -r requirements.txt

--------------------------------------------------------------------
📷 CAMERA SETUP
--------------------------------------------------------------------
Update camera source in code:

Local webcam:
> cap = cv2.VideoCapture(0)

DroidCam / IP Webcam:
> CAMERA_SOURCE = "http://<your-ip>:4747/video"

--------------------------------------------------------------------
🚀 HOW TO RUN
--------------------------------------------------------------------
1. Connect camera
2. Run main script:
> python hand.py
3. Select mode via UI buttons or keyboard keys
4. Start interacting with your hand gestures 🎯

--------------------------------------------------------------------
📌 FILE STRUCTURE
--------------------------------------------------------------------
hand.py                → Main project code
README.txt             → Documentation (this file)
requirements.txt       → Dependencies list

--------------------------------------------------------------------
📜 LICENSE
--------------------------------------------------------------------
This project is for educational and research purposes only.  
Use responsibly.

--------------------------------------------------------------------
👤 AUTHOR
--------------------------------------------------------------------
Developed by: **Harshit Shaw**


--------------------------------------------------------------------
