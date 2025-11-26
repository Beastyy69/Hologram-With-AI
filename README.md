============================================================
        AR + Mouse + AI Shapes (Gesture Control System)
============================================================

Project Type : Computer Vision + Human-Computer Interaction  
Language     : Python 3  
Frameworks   : OpenCV, MediaPipe, AI (Optional)

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------
This software enables touchless interaction with a computer
using real-time hand tracking through a webcam.

Core Functionalities:
- Gesture-controlled 3D AR object manipulation
- Touchless mouse control using hand gestures
- Air drawing using a single finger
- Voice controlled AI shape generation (optional)

Ideal for:
- Educational Projects
- Research on Natural User Interfaces
- Computer Vision + AR Experiments

------------------------------------------------------------
MAIN FEATURES
------------------------------------------------------------
1️⃣ 3D AR SHAPE MODE  
   • Rotate, Scale, Move 3D objects using two hands

2️⃣ AI SHAPE MODE (Optional)  
   • Create shapes from spoken prompts
   • Example: "Letter A", "Number 3", "Triangle"

3️⃣ MOUSE CONTROL MODE  
   • Index finger → Move cursor  
   • Pinch → Left click  
   • Pinch with middle → Right click  
   • Closed fist → Drag + Drop  

4️⃣ AIR DRAWING MODE  
   • Draw in the air using index finger
   • Choose colors + Clear screen

------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------
Hardware:
• Webcam (HD recommended)
• Microphone (only for voice mode)

Python Libraries:
• opencv-python
• mediapipe
• numpy
• pynput
• pyautogui
• google-generativeai (optional)
• speechrecognition + pyaudio (optional)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
Step 1: Install dependencies:
    pip install -r requirements.txt

Step 2: Run the software:
    python main.py

The webcam window will open with live interaction.

------------------------------------------------------------
CONTROLS AND SHORTCUTS
------------------------------------------------------------
Keys:
• 1 → 3D AR Shape Mode
• 2 → AI Shape Mode
• 3 → Mouse Mode
• 4 → Air Drawing Mode
• A → Auto-rotate ON/OFF
• Q → Quit program

------------------------------------------------------------
OPTIONAL AI + VOICE SETUP
------------------------------------------------------------
For security reasons the public version disables:
- Google Gemini API Key
- Voice Recognition

To enable them:
1. Open the code
2. Add your own Gemini API Key
3. Set:
     GEMINI_AVAILABLE = True
     SR_AVAILABLE = True
4. Uncomment voice listener thread in main()

------------------------------------------------------------
PERFORMANCE TIPS
------------------------------------------------------------
• Ensure good lighting so hands are detected correctly  
• Keep hand in frame  
• Use plain background for better tracking stability  

------------------------------------------------------------
AUTHOR DETAILS
------------------------------------------------------------
Developed by : Harshit Shaw  
Project Topic: Gesture-Based Human-Computer Interaction  
Version      : Public Release 1.0

------------------------------------------------------------
LICENSE & USAGE POLICY
------------------------------------------------------------
This project is provided for EDUCATIONAL and DEMONSTRATION
purposes only.

Redistribution without proper credit is prohibited.
Commercial usage requires owner permission.

============================================================
            END OF DOCUMENT – THANK YOU
============================================================
