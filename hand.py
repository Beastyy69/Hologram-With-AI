import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from pynput.mouse import Button, Controller
import pyautogui
import threading
import queue
import time
import json
import re
#Add All The Variables Here
# ===== OPTIONAL for: Gemini + Speech Recognition =====
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    GENAI_API_KEY = "GEMINI_API_KEY_HERE"  # Replace with your actual Gemini API key
    if GENAI_API_KEY and GENAI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=GENAI_API_KEY)
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False


# ===== CAMERA CONFIG =====
CAMERA_SOURCE = "0"  # Change to video file path for pre-recorded video
current_mode = "CUBE"
auto_rotate = False

# ===== MediaPipe Hands =====
mpHands = mp.solutions.hands
draw = mp.solutions.drawing_utils

MP_MODEL_COMPLEXITY = 0
MP_MIN_DETECTION_CONF = 0.55
MP_MIN_TRACKING_CONF = 0.7
MP_INPUT_SCALE = 0.75

mouse = Controller()
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
TRACKPAD_W = 0.60  
TRACKPAD_H = 0.60  
CURSOR_DEADZONE_PX = 2.5
pos_history = deque(maxlen=6)
left_clicked = False
right_clicked = False
dragging = False

shape_pos = [320, 240]
shape_scale = 40.0
rot_x, rot_y, rot_z = 0.0, 0.0, 0.0

last_ai_command = ""
last_ai_status = ""
BUTTONS = {}
command_queue = queue.Queue()

# ===== Built-in Shapes + Geometric Primitives =====
cube_vertices_base = np.float32([
    [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
    [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]
])
cube_edges_base = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]
pyramid_vertices_base = np.float32([
    [-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1],[ 0, 1, 0]
])
pyramid_edges_base = [
    (0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)
]
sphere_vertices_base = np.float32([
    [0,0,1],[0.894,0,0.447],[0.276,0.851,0.447],[-0.724,0.526,0.447],[-0.724,-0.526,0.447],[0.276,-0.851,0.447],
    [0.724,0.526,-0.447],[-0.276,0.851,-0.447],[-0.894,0,-0.447],[-0.276,-0.851,-0.447],[0.724,-0.526,-0.447],[0,0,-1]
])
sphere_edges_base = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,1),
    (1,6),(2,6),(2,7),(3,7),(3,8),(4,8),(4,9),(5,9),(5,10),(1,10),
    (6,7),(7,8),(8,9),(9,10),(10,6),(6,11),(7,11),(8,11),(9,11),(10,11)
]

def create_polygon_prism(n_sides, extrude_depth=0.6):
    """Create a 3D prism with n-sided polygon base"""
    angles = np.linspace(0, 2*math.pi, n_sides, endpoint=False)
    
    front = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.full(n_sides, -extrude_depth/2)
    ])
    
    back = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.full(n_sides, extrude_depth/2)
    ])
    
    vertices = np.vstack([front, back]).astype(np.float32)
    
    edges = []
    for i in range(n_sides):
        edges.append((i, (i+1) % n_sides))
    
    for i in range(n_sides):
        edges.append((n_sides + i, n_sides + (i+1) % n_sides))
    
    for i in range(n_sides):
        edges.append((i, n_sides + i))
    
    return vertices, edges

triangle_vertices_base, triangle_edges_base = create_polygon_prism(3)
pentagon_vertices_base, pentagon_edges_base = create_polygon_prism(5)
hexagon_vertices_base, hexagon_edges_base = create_polygon_prism(6)
octagon_vertices_base, octagon_edges_base = create_polygon_prism(8)

rhombus_vertices_base = np.float32([
    [0, 1, -0.3], [1, 0, -0.3], [0, -1, -0.3], [-1, 0, -0.3],  
    [0, 1, 0.3], [1, 0, 0.3], [0, -1, 0.3], [-1, 0, 0.3] 
])
rhombus_edges_base = [
    (0,1),(1,2),(2,3),(3,0),  
    (4,5),(5,6),(6,7),(7,4),  
    (0,4),(1,5),(2,6),(3,7)   
]

shape_vertices = cube_vertices_base.copy()
shape_edges = cube_edges_base.copy()
current_shape_name = "Cube"

# ===== Kalman Filter =====
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2) * 10
        self.Q = np.eye(4) * 0.1

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        measurement = np.array(measurement)
        _ = self.predict()
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]

mouse_filter = KalmanFilter()


class CursorStabilizer:
    """Kalman + moving average + adaptive EMA with deadzone for low-jitter cursor control."""
    def __init__(self, ma_window=4, slow_alpha=0.2, fast_alpha=0.65, speed_norm=85.0, deadzone_px=2.5):
        self.filter = KalmanFilter()
        self.history = deque(maxlen=ma_window)
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.speed_norm = speed_norm
        self.deadzone_px = deadzone_px
        self.ema = None
        self.last_output = None

    def _alpha(self, speed):
        ratio = np.clip(speed / self.speed_norm, 0.0, 1.0)
        return self.slow_alpha + (self.fast_alpha - self.slow_alpha) * ratio

    def update(self, measured_xy):
        kalman_xy = self.filter.update(measured_xy)
        self.history.append(np.array(kalman_xy, dtype=np.float32))
        avg_xy = np.mean(self.history, axis=0)

        if self.ema is None:
            self.ema = avg_xy.copy()

        speed = 0.0
        if self.last_output is not None:
            speed = float(np.linalg.norm(avg_xy - self.last_output))

        alpha = self._alpha(speed)
        self.ema = (1.0 - alpha) * self.ema + alpha * avg_xy

        if self.last_output is not None:
            if np.linalg.norm(self.ema - self.last_output) < self.deadzone_px:
                self.ema = self.last_output.copy()

        self.ema[0] = np.clip(self.ema[0], 0, screen_w)
        self.ema[1] = np.clip(self.ema[1], 0, screen_h)
        self.last_output = self.ema.copy()
        return float(self.ema[0]), float(self.ema[1])


cursor_stabilizer = CursorStabilizer(deadzone_px=CURSOR_DEADZONE_PX)


class HandPipeline:
    """Mode-aware lightweight MediaPipe runtime to reduce per-frame latency."""
    def __init__(self):
        self.single_hand = mpHands.Hands(
            max_num_hands=1,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )
        self.dual_hand = mpHands.Hands(
            max_num_hands=2,
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

    def process(self, frame_bgr, mode):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if MP_INPUT_SCALE < 1.0:
            rgb = cv2.resize(rgb, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv2.INTER_LINEAR)
        rgb.flags.writeable = False
        if mode in ("MOUSE", "DRAW"):
            return self.single_hand.process(rgb)
        return self.dual_hand.process(rgb)

    def close(self):
        self.single_hand.close()
        self.dual_hand.close()


hand_pipeline = HandPipeline()

# ===== AR Interaction Tuning =====
# Base smoothing removes jitter; fast smoothing keeps quick gestures responsive.
AR_BASE_ALPHA = 0.18
AR_FAST_ALPHA = 0.58
AR_SPEED_NORM = 28.0


class MotionAverager:
    """Simple moving average for scalar or 2D points."""
    def __init__(self, window=4):
        self.values = deque(maxlen=window)

    def update(self, value):
        arr = np.asarray(value, dtype=np.float32)
        self.values.append(arr)
        return np.mean(self.values, axis=0)


class ScalarKalmanFilter:
    """1D constant-velocity Kalman filter used for smooth scale/rotation."""
    def __init__(self, process_var=0.02, measure_var=5.0):
        self.state = np.zeros(2, dtype=np.float32)  # [value, velocity]
        self.P = np.eye(2, dtype=np.float32) * 1000.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)
        self.Q = np.eye(2, dtype=np.float32) * process_var
        self.R = np.array([[measure_var]], dtype=np.float32)

    def update(self, measurement):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        measurement_vec = np.array([float(measurement)], dtype=np.float32)
        y = measurement_vec - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        self.P = (np.eye(2, dtype=np.float32) - (K @ self.H)) @ self.P
        return float(self.state[0])


class ARGestureStabilizer:
    """Stabilizes AR move/rotate/scale while preserving gesture mapping behavior."""
    def __init__(self):
        self.hand_center_filters = {}
        self.midpoint_filter = KalmanFilter()
        self.scale_filter = ScalarKalmanFilter(process_var=0.08, measure_var=20.0)
        self.rot_x_filter = ScalarKalmanFilter(process_var=0.04, measure_var=2.0)
        self.rot_y_filter = ScalarKalmanFilter(process_var=0.04, measure_var=2.0)
        self.pos_averager = MotionAverager(window=3)
        self.prev_two_hand_dist = None
        self.prev_midpoint = None

    @staticmethod
    def _adaptive_alpha(speed):
        speed_ratio = min(max(speed / AR_SPEED_NORM, 0.0), 1.0)
        return AR_BASE_ALPHA + (AR_FAST_ALPHA - AR_BASE_ALPHA) * speed_ratio

    def get_hand_center(self, hand_idx, palm_center):
        if hand_idx not in self.hand_center_filters:
            self.hand_center_filters[hand_idx] = KalmanFilter()
        return self.hand_center_filters[hand_idx].update(palm_center)

    def smooth_scale_rotation(self, lm1, f1, idx1, lm2, f2, idx2, w, h, current_scale, current_rx, current_ry):
        c1 = self.get_hand_center(idx1, lm1[9])
        c2 = self.get_hand_center(idx2, lm2[9])
        midpoint = (c1 + c2) / 2.0
        dist = np.linalg.norm(c1 - c2)

        open1 = sum(f1) >= 3
        open2 = sum(f2) >= 3
        if not (open1 and open2):
            self.prev_two_hand_dist = dist
            return None

        scale_delta = 0.0
        if self.prev_two_hand_dist is not None:
            scale_delta = dist - self.prev_two_hand_dist
        self.prev_two_hand_dist = dist

        target_scale = np.clip(current_scale + scale_delta, 10, 600)
        kf_scale = self.scale_filter.update(target_scale)
        scale_alpha = self._adaptive_alpha(abs(scale_delta))
        new_scale = (1.0 - scale_alpha) * current_scale + scale_alpha * kf_scale

        dx = (midpoint[0] - w / 2.0) / (w / 2.0)
        dy = (midpoint[1] - h / 2.0) / (h / 2.0)
        target_ry = dx * math.pi * 1.2
        target_rx = dy * math.pi * 1.2

        kf_rx = self.rot_x_filter.update(target_rx)
        kf_ry = self.rot_y_filter.update(target_ry)
        rot_speed = max(abs(kf_rx - current_rx), abs(kf_ry - current_ry)) * 30.0
        rot_alpha = self._adaptive_alpha(rot_speed)
        new_rx = (1.0 - rot_alpha) * current_rx + rot_alpha * kf_rx
        new_ry = (1.0 - rot_alpha) * current_ry + rot_alpha * kf_ry

        return float(new_scale), float(new_rx), float(new_ry)

    def smooth_position(self, target_point, current_pos):
        kf_point = self.midpoint_filter.update(target_point)
        avg_point = self.pos_averager.update(kf_point)

        speed = 0.0
        if self.prev_midpoint is not None:
            speed = float(np.linalg.norm(avg_point - self.prev_midpoint))
        self.prev_midpoint = np.array(avg_point, dtype=np.float32)

        pos_alpha = self._adaptive_alpha(speed)
        new_x = (1.0 - pos_alpha) * current_pos[0] + pos_alpha * avg_point[0]
        new_y = (1.0 - pos_alpha) * current_pos[1] + pos_alpha * avg_point[1]
        return [float(new_x), float(new_y)]


ar_stabilizer = ARGestureStabilizer()

# ===== Helper Functions =====
def rotation_matrix(rx, ry, rz):
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def get_hand_side_from_mh(mh):
    return mh.classification[0].label == "Left"

def finger_states(lm, is_left):
    fingers=[0]*5
    if is_left:
        fingers[0] = 1 if lm[4][0] > lm[3][0]+20 else 0
    else:
        fingers[0] = 1 if lm[4][0] < lm[3][0]-20 else 0
    tips=[8,12,16,20]; dips=[6,10,14,18]
    for i,(t,d) in enumerate(zip(tips,dips),start=1):
        fingers[i] = 1 if lm[t][1] < lm[d][1]-15 else 0
    return fingers

def adaptive_pinch_threshold(lm):
    hand_size = np.linalg.norm(np.array(lm[0])-np.array(lm[9]))
    return max(20,min(80,hand_size*0.15))

def is_pinch(p1,p2,lm):
    return np.linalg.norm(np.array(p1)-np.array(p2)) < adaptive_pinch_threshold(lm)

# Drawing Shape 
def draw_shape(frame):
    global shape_vertices, shape_edges, rot_x, rot_y, rot_z
    if shape_vertices is None or len(shape_vertices) == 0:
        return
    R = rotation_matrix(rot_x,rot_y,rot_z)
    v = (shape_vertices @ R.T) * shape_scale
    v[:,1] *= -1  

    v[:,0] += shape_pos[0]
    v[:,1] += shape_pos[1]

    v = v.astype(int)
    for e in shape_edges:
        a,b = e
        if a < 0 or b < 0 or a >= len(v) or b >= len(v): continue
        cv2.line(frame, tuple(v[a][:2]), tuple(v[b][:2]), (0,255,0), 2)

# ====== AR GESTURES (FIXED - NO RESET) ======
def handle_ar_gestures(frame, results):
    global shape_pos, shape_scale, rot_x, rot_y
    h, w = frame.shape[:2]

    if not results.multi_hand_landmarks:
        return

    hands_info = []
    for i, hand in enumerate(results.multi_hand_landmarks):
        lm = [(int(l.x*w), int(l.y*h)) for l in hand.landmark]
        mh = results.multi_handedness[i]
        is_left = get_hand_side_from_mh(mh)
        fingers = finger_states(lm, is_left)
        hands_info.append((lm, fingers, i))

    if len(hands_info) == 2:
        (lm1, f1, idx1), (lm2, f2, idx2) = hands_info

        open1 = sum(f1) >= 3
        open2 = sum(f2) >= 3
        index1 = (f1 == [0,1,0,0,0])
        index2 = (f2 == [0,1,0,0,0])

        if open1 and open2:
            stabilized = ar_stabilizer.smooth_scale_rotation(
                lm1, f1, idx1, lm2, f2, idx2, w, h, shape_scale, rot_x, rot_y
            )
            if stabilized is not None:
                shape_scale, rot_x, rot_y = stabilized

        elif index1 and index2:
            c1 = ar_stabilizer.get_hand_center(idx1, lm1[9])
            c2 = ar_stabilizer.get_hand_center(idx2, lm2[9])
            midpoint = (c1 + c2) / 2.0
            shape_pos = ar_stabilizer.smooth_position(midpoint, shape_pos)

        else:
            c1 = ar_stabilizer.get_hand_center(idx1, lm1[9])
            c2 = ar_stabilizer.get_hand_center(idx2, lm2[9])
            ar_stabilizer.prev_two_hand_dist = np.linalg.norm(c1 - c2)

    else:
        (lm, f, _) = hands_info[0]
        if f == [0,1,0,0,0]:
            shape_pos = ar_stabilizer.smooth_position(lm[9], shape_pos)

# ===== Mouse Mode =====
def smooth_cursor(x, y, w, h):
    # Define trackpad region (bottom-right corner)
    pad_left = int((w - (w * TRACKPAD_W)) / 2)
    pad_top = int((h - (h * TRACKPAD_H)) / 2)
    pad_right = int((w + (w * TRACKPAD_W)) / 2)
    pad_bottom = int((h + (h * TRACKPAD_H)) / 2)


    if not (pad_left <= x <= pad_right and pad_top <= y <= pad_bottom):
        return

    nx = (x - pad_left) / (pad_right - pad_left)
    ny = (y - pad_top) / (pad_bottom - pad_top)

    sx = nx * screen_w
    sy = ny * screen_h
    stabilized = cursor_stabilizer.update((sx, sy))
    mouse.position = stabilized

def detect_mouse_gesture(frame,lm,w,h,is_left):
    global left_clicked,right_clicked,dragging

    fingers=finger_states(lm,is_left)
    pinch_idx=is_pinch(lm[8],lm[4],lm)
    pinch_mid=is_pinch(lm[12],lm[4],lm)

    if fingers==[0,0,0,0,0]:
        if not dragging:
            dragging=True
            mouse.press(Button.left)
        smooth_cursor(lm[8][0], lm[8][1], w, h)
        return

    if dragging and fingers==[1,1,1,1,1]:
        mouse.release(Button.left)
        dragging=False
        return

    global last_click_time

    if pinch_idx and not pinch_mid:
        if time.time() - last_click_time > CLICK_COOLDOWN:
            mouse.click(Button.left)
            last_click_time = time.time()

        else:
            left_clicked=False

    if pinch_idx and pinch_mid:
        if not right_clicked:
            right_clicked=True
            mouse.click(Button.right)
    else:
        right_clicked=False

    if fingers==[0,1,0,0,0]:
        smooth_cursor(lm[8][0],lm[8][1],w,h)

def handle_mouse_mode(frame,results,w,h):
    if not results.multi_hand_landmarks: return
    hand=results.multi_hand_landmarks[0]
    mh=results.multi_handedness[0]
    is_left=get_hand_side_from_mh(mh)
    lm=[(int(l.x*w),int(l.y*h)) for l in hand.landmark]
    detect_mouse_gesture(frame,lm,w,h,is_left)
    draw.draw_landmarks(frame,hand,mpHands.HAND_CONNECTIONS)

# ===== IMPROVED Letter/Number Template Generation (3D EXTRUSION) =====
def resample_contour_pts(pts, n):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], n, axis=0)
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    d = np.sqrt((diffs**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (n,1))
    t = np.linspace(0, total, n, endpoint=False)
    res = []
    for ti in t:
        idx = np.searchsorted(cum, ti, side='right') - 1
        idx = np.clip(idx, 0, len(pts)-1)
        seg_len = d[idx]
        if seg_len == 0:
            res.append(pts[idx])
            continue
        local_t = (ti - cum[idx]) / seg_len
        p0 = pts[idx]
        p1 = pts[(idx+1) % len(pts)]
        res.append((1-local_t)*p0 + local_t*p1)
    return np.array(res, dtype=np.float32)

def all_contours_from_char(char, img_size=800, font_scale=10.0, thickness=15):
    """Extract ALL contours (outer + inner holes) for complete letter representation"""
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(char)
    ((tw, th), _) = cv2.getTextSize(text, font, font_scale, thickness)
    org = (img_size//2 - tw//2, img_size//2 + th//2)
    cv2.putText(canvas, text, org, font, font_scale, 255, thickness, lineType=cv2.LINE_AA)
    
    canvas = cv2.GaussianBlur(canvas, (7,7), 1.5)
    _, bw = cv2.threshold(canvas, 30, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    valid_contours = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 100: 
            cnt_squeezed = cnt.squeeze()
            if cnt_squeezed.ndim == 1:
                cnt_squeezed = cnt_squeezed[np.newaxis, :]
            valid_contours.append(cnt_squeezed.astype(np.float32))
    
    return valid_contours if valid_contours else None

def shape_wireframe_from_char_3d(char, samples_per_contour=60, extrude_depth=0.4):
    """Create proper 3D extruded wireframe with front and back faces"""
    contours = all_contours_from_char(char)
    if contours is None or len(contours) == 0:
        return None, None
    
    all_verts = []
    all_edges = []
    vertex_offset = 0
    
    all_pts = np.vstack(contours)
    minxy = all_pts.min(axis=0)
    maxxy = all_pts.max(axis=0)
    ctr = (minxy + maxxy) / 2.0
    scale = max((maxxy - minxy)[0], (maxxy - minxy)[1])
    if scale == 0:
        scale = 1.0
    
    for contour in contours:
        pts = resample_contour_pts(contour, samples_per_contour)
        
        norm = (pts - ctr) / (scale / 2.0)
        norm[:,1] *= -1.0  
        
        n = len(norm)
        
        front_verts = np.hstack([norm, np.full((n, 1), -extrude_depth/2)])
        
        back_verts = np.hstack([norm, np.full((n, 1), extrude_depth/2)])
        
        verts = np.vstack([front_verts, back_verts])
        all_verts.append(verts)
        
        for i in range(n):
            all_edges.append((vertex_offset + i, vertex_offset + (i+1)%n))
        

        for i in range(n):
            all_edges.append((vertex_offset + n + i, vertex_offset + n + (i+1)%n))
        

        for i in range(0, n, 4):
            all_edges.append((vertex_offset + i, vertex_offset + n + i))
        
        vertex_offset += 2 * n
    
    final_verts = np.vstack(all_verts).astype(np.float32)
    
    return final_verts, all_edges

number_shapes = {}
letter_shapes = {}

def build_wireframe_templates(samples=60, depth=0.4):
    #Declare global variables
    
    for ch in "0123456789":
        v, e = shape_wireframe_from_char_3d(ch, samples_per_contour=samples, extrude_depth=depth)
        if v is not None:
            number_shapes[ch] = (v.astype(np.float32), e)
    
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        v, e = shape_wireframe_from_char_3d(ch, samples_per_contour=samples, extrude_depth=depth)
        if v is not None:
            letter_shapes[ch] = (v.astype(np.float32), e)

try:
    build_wireframe_templates(samples=60, depth=0.5)
    print(f"Built {len(number_shapes)} numbers and {len(letter_shapes)} letters")
except Exception as e:
    print(f"Template build error: {e}")

def extract_json_from_text(raw_text):
    txt = raw_text.replace("```json","").replace("```","")
    txt = re.sub(r"//.*", "", txt)
    m = re.search(r"\{[\s\S]*?\}", txt)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def set_builtin_shape(name):
    #Declare global variables
    name=name.lower()
    if "cube" in name:
        shape_vertices=cube_vertices_base.copy()
        shape_edges=cube_edges_base.copy()
        current_shape_name="Cube"; last_ai_status="✓ Builtin Cube"; return True
    if "pyramid" in name:
        shape_vertices=pyramid_vertices_base.copy()
        shape_edges=pyramid_edges_base.copy()
        current_shape_name="Pyramid"; last_ai_status="✓ Builtin Pyramid"; return True
    if "sphere" in name or "ball" in name:
        shape_vertices=sphere_vertices_base.copy()
        shape_edges=sphere_edges_base.copy()
        current_shape_name="Sphere"; last_ai_status="✓ Builtin Sphere"; return True
    if "rhombus" in name or "diamond" in name:
        shape_vertices=rhombus_vertices_base.copy()
        shape_edges=rhombus_edges_base.copy()
        current_shape_name="Rhombus"; last_ai_status="✓ Builtin Rhombus"; return True
    if "triangle" in name:
        shape_vertices=triangle_vertices_base.copy()
        shape_edges=triangle_edges_base.copy()
        current_shape_name="Triangle"; last_ai_status="✓ Builtin Triangle"; return True
    if "pentagon" in name:
        shape_vertices=pentagon_vertices_base.copy()
        shape_edges=pentagon_edges_base.copy()
        current_shape_name="Pentagon"; last_ai_status="✓ Builtin Pentagon"; return True
    if "hexagon" in name:
        shape_vertices=hexagon_vertices_base.copy()
        shape_edges=hexagon_edges_base.copy()
        current_shape_name="Hexagon"; last_ai_status="✓ Builtin Hexagon"; return True
    if "octagon" in name:
        shape_vertices=octagon_vertices_base.copy()
        shape_edges=octagon_edges_base.copy()
        current_shape_name="Octagon"; last_ai_status="✓ Builtin Octagon"; return True
    return False

def generate_shape_from_text(text):
    #Declare global variables
    last_ai_command = text.strip()
    t = last_ai_command.strip().lower()

    letter_match = re.search(r'\b(?:letter|alphabet)\s+([a-z])\b', t)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in letter_shapes:
            shape_vertices, shape_edges = letter_shapes[letter]
            current_shape_name = f"Letter {letter}"
            last_ai_status = "✓ Template letter"
            current_mode = "CUBE"
            return
    
    number_match = re.search(r'\b(?:number|digit)\s+(\d|zero|one|two|three|four|five|six|seven|eight|nine)\b', t)
    if number_match:
        num_str = number_match.group(1)

        word_to_digit = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
                         "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        digit = word_to_digit.get(num_str, num_str)
        
        if digit in number_shapes:
            shape_vertices, shape_edges = number_shapes[digit]
            current_shape_name = f"Number {digit}"
            last_ai_status = "✓ Template number"
            current_mode = "CUBE"
            return

    if len(t) == 1:
        if t.isdigit() and t in number_shapes:
            shape_vertices, shape_edges = number_shapes[t]
            current_shape_name = f"Number {t}"
            last_ai_status = "✓ Template number"
            current_mode = "CUBE"
            return
        k = t.upper()
        if k in letter_shapes:
            shape_vertices, shape_edges = letter_shapes[k]
            current_shape_name = f"Letter {k}"
            last_ai_status = "✓ Template letter"
            current_mode = "CUBE"
            return
    
    word_to_digit = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
                     "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
    if t in word_to_digit:
        d = word_to_digit[t]
        if d in number_shapes:
            shape_vertices, shape_edges = number_shapes[d]
            current_shape_name = f"Number {d}"
            last_ai_status = "✓ Template (word)"
            current_mode = "CUBE"
            return

    if set_builtin_shape(t):
        current_mode = "CUBE"
        return

    if not GEMINI_AVAILABLE:
        shape_vertices = cube_vertices_base.copy()
        shape_edges = cube_edges_base.copy()
        current_shape_name = "Cube"
        last_ai_status = "Gemini OFF"
        current_mode = "CUBE"
        return

    try:
        #Add The Prompt To Generate The 3D Shape Here
        prompt = f""" """
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        data = extract_json_from_text(resp.text if hasattr(resp, "text") else str(resp))
        if data is None:
            raise ValueError("No JSON")

        verts = np.array(data["vertices"], dtype=np.float32)
        edges = [tuple(e) for e in data["edges"]]

        if verts.ndim == 2 and verts.shape[1] >= 3 and np.allclose(verts[:,2], 0.0):
            verts[:,2] = np.linspace(-0.5, 0.5, verts.shape[0])

        max_dim = np.max(np.abs(verts))
        if max_dim > 0:
            verts = verts / max(max_dim, 1.0)

        shape_vertices = verts.astype(np.float32)
        shape_edges = edges
        current_shape_name = t
        last_ai_status = "✓ AI generated"
        current_mode = "CUBE"
    except Exception as e:
        last_ai_status = f"AI error: {str(e)[:30]}"
        shape_vertices = pyramid_vertices_base.copy()
        shape_edges = pyramid_edges_base.copy()
        current_shape_name = "Pyramid"
        current_mode = "CUBE"


def voice_listener():
    if not SR_AVAILABLE: return
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,1)
        while True:
            try:
                audio = r.listen(source, timeout=2, phrase_time_limit=4)
                text = r.recognize_google(audio).lower()
                if current_mode == "AI":
                    command_queue.put(text)
            except:
                pass

def draw_mode_buttons(frame):
    global BUTTONS
    h,w = frame.shape[:2]
    BUTTONS = {
   # Mode Buttons Add Here
    }
    for mode,((x1,y1),(x2,y2)) in BUTTONS.items():
        col=(0,255,0) if mode==current_mode else (100,100,100)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        cv2.putText(frame,mode,(x1+10,y1+25),cv2.FONT_HERSHEY_PLAIN,1.6,col,2)

    if current_mode != "DRAW":
        return

    cv2.rectangle(frame,(40, 1),(140,65),(0,0,0),-1)
    cv2.putText(frame,"CLEAR",(52,38),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    btns = [(160,255,"BLUE"),(255,350,"GREEN"),(350,445,"RED"),(445,540,"YELLOW")]
    for i,(x1,x2,label) in enumerate(btns):
        cv2.rectangle(frame,(x1,1),(x2,65),color[i],-1)
        cv2.putText(frame,label,(x1+10,38),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)



def on_mouse(event,x,y,flags,param):
    global current_mode
    if event==cv2.EVENT_LBUTTONDOWN:
        for mode,((x1,y1),(x2,y2)) in BUTTONS.items():
            if x1<=x<=x2 and y1<=y<=y2:
                current_mode=mode
                break
def handle_air_draw_mode(frame, results, w, h):
    global bpoints,gpoints,rpoints,ypoints
    global blue_index,green_index,red_index,yellow_index,colorIndex

    if not results.multi_hand_landmarks:
        return

    landmarks = results.multi_hand_landmarks[0]
    lm = [(int(pt.x*w), int(pt.y*h)) for pt in landmarks.landmark]
    draw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)

    hand_id = 0  # Only first hand used for drawing
    if hand_id not in drawing_filters:
        drawing_filters[hand_id] = KalmanFilter()
    raw_pt = lm[8]
    smooth_pt = drawing_filters[hand_id].update(raw_pt)
    index_tip = (int(smooth_pt[0]), int(smooth_pt[1]))
    middle_tip = lm[12]
    index_pip, middle_pip = lm[6], lm[10]

    index_up = index_tip[1] < index_pip[1]
    middle_up = middle_tip[1] < middle_pip[1]

    drawing_mode = index_up and not middle_up

    cv2.circle(frame,index_tip,7,(0,0,0),-1)

    if index_tip[1] <= 65:
        if 40<=index_tip[0]<=140:
            bpoints=[deque(maxlen=1024)]
            gpoints=[deque(maxlen=1024)]
            rpoints=[deque(maxlen=1024)]
            ypoints=[deque(maxlen=1024)]
            blue_index=green_index=red_index=yellow_index=0
        elif 160<=index_tip[0]<=255: colorIndex=0
        elif 275<=index_tip[0]<=370: colorIndex=1
        elif 390<=index_tip[0]<=485: colorIndex=2
        elif 505<=index_tip[0]<=600: colorIndex=3

    elif drawing_mode:
        if colorIndex==0: bpoints[blue_index].appendleft(index_tip)
        elif colorIndex==1: gpoints[green_index].appendleft(index_tip)
        elif colorIndex==2: rpoints[red_index].appendleft(index_tip)
        elif colorIndex==3: ypoints[yellow_index].appendleft(index_tip)
    else:
        bpoints.append(deque(maxlen=1024))
        gpoints.append(deque(maxlen=1024))
        rpoints.append(deque(maxlen=1024))
        ypoints.append(deque(maxlen=1024))
        blue_index+=1; green_index+=1; red_index+=1; yellow_index+=1

    points=[bpoints,gpoints,rpoints,ypoints]
    for i in range(4):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k] and points[i][j][k-1]:
                    cv2.line(frame,points[i][j][k-1],points[i][j][k],color[i],5)

def handle_drawing_mode(frame, results, w, h):
    global drawing_mode
    
    if not results.multi_hand_landmarks:
        return
    
    hand = results.multi_hand_landmarks[0]
    lm = [(int(l.x * w), int(l.y * h)) for l in hand.landmark]
    is_left = get_hand_side_from_mh(results.multi_handedness[0])
    fingers = finger_states(lm, is_left)

    index_up = (fingers == [0,1,0,0,0])
    fist = (fingers == [0,0,0,0,0])

    if index_up:
        drawing_mode = True
        draw_points.append(lm[8])  
    elif fist:
        drawing_mode = False
        draw_points.append(None)  


    for i in range(1, len(draw_points)):
        if draw_points[i] is not None and draw_points[i-1] is not None:
            cv2.line(frame, draw_points[i-1], draw_points[i], (0,0,255), 4)

    draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)


def main():
    #Declare global variables
    drawing_mode = False

    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Camera not opened.")
        return

    cv2.namedWindow("AR + Mouse + AI Shapes")
    cv2.setMouseCallback("AR + Mouse + AI Shapes", on_mouse)

    if SR_AVAILABLE:
        threading.Thread(target=voice_listener,daemon=True).start()

    print("\n=== CONTROLS ===")
    print("1/2/3 - Switch modes")
    print("A - Toggle auto-rotate")
    print("Q - Quit")
    print("\n=== VOICE COMMANDS (AI Mode) ===")
    print("LETTERS:")
    print("  'letter A', 'letter B', 'alphabet C', etc.")
    print("  OR just 'A', 'B', 'C', etc.")
    print("\nNUMBERS:")
    print("  'number 5', 'number 3', 'digit 7', etc.")
    print("  'number five', 'number three', etc.")
    print("  OR just '5', '3', 'five', 'three', etc.")
    print("\nSHAPES:")
    print("  'triangle', 'pentagon', 'hexagon', 'octagon'")
    print("  'rhombus', 'diamond', 'cube', 'pyramid', 'sphere'")
    print("\nCUSTOM:")
    print("  Any shape description for AI generation")
    print("\n=== GESTURES ===")
    print("Two hands open: Scale & rotate shape")
    print("Two index fingers: Move shape")
    print("One index finger: Move shape\n")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape
        results = None
        if current_mode in ("CUBE", "AI", "MOUSE", "DRAW"):
            results = hand_pipeline.process(frame, current_mode)

        if current_mode=="AI":
            while not command_queue.empty():
                generate_shape_from_text(command_queue.get())

        if current_mode in ("CUBE","AI"):
            if auto_rotate: rot_y += 0.015
            draw_shape(frame)
            if results and results.multi_hand_landmarks:
                handle_ar_gestures(frame, results)
                for hand in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

        elif current_mode=="MOUSE":
            pad_left = int((w - (w * TRACKPAD_W)) / 2)
            pad_top = int((h - (h * TRACKPAD_H)) / 2)
            pad_right = int((w + (w * TRACKPAD_W)) / 2)
            pad_bottom = int((h + (h * TRACKPAD_H)) / 2)
            cv2.rectangle(frame, (pad_left, pad_top), (pad_right, pad_bottom), (255, 255, 255), 2)
            cv2.putText(frame, "TRACKPAD (move index finger)",
            (pad_left + 10, pad_top + 20),
            cv2.FONT_HERSHEY_PLAIN, 1.1, (255,255,255), 2)
            if results and results.multi_hand_landmarks:
                handle_mouse_mode(frame, results, w, h)
        elif current_mode == "DRAW":
            if results and results.multi_hand_landmarks:
                handle_air_draw_mode(frame,results,w,h)


        cv2.putText(frame, f"Mode: {current_mode}", (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(frame, f"Shape: {current_shape_name}", (10,55), cv2.FONT_HERSHEY_PLAIN, 1.4, (0,255,255), 2)
        if last_ai_command:
            cv2.putText(frame, f"Voice: {last_ai_command}", (10,80), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,0), 2)
        if last_ai_status:
            cv2.putText(frame, last_ai_status, (10,105), cv2.FONT_HERSHEY_PLAIN, 1.2, (200,200,200), 2)

        draw_mode_buttons(frame)

        cv2.imshow("AR + Mouse + AI Shapes", frame)
    #Make Cases For Perfoming Which Mode Here

    cap.release()
    hand_pipeline.close()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
