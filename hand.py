import cv2
import numpy as np
import math
import os
from collections import deque
from pynput.mouse import Button, Controller
from pynput import keyboard as kb
import pyautogui
import threading
import queue
import time
import json
import re
import urllib.request

BUILD_TAG = "v7 (curriculum 3D models, decluttered UI, improved voice pickup)"
print(f"[HOLOGRAM BUILD] {BUILD_TAG}")

# ===== Hand Detection Setup (with MediaPipe fallback) =====
MEDIAPIPE_OK = False
USE_MEDIAPIPE = True
draw = None
hands = None
mpHands = None

try:
    import mediapipe as mp
    # Try to import solutions
    try:
        # Try standard approach first
        if hasattr(mp, 'solutions'):
            from mediapipe import solutions
            mpHands = solutions.hands
            draw = solutions.drawing_utils
            MEDIAPIPE_OK = True
        else:
            raise AttributeError("mp.solutions not available")
    except:
        # Try direct import
        from mediapipe.solutions import hands as hands_module
        from mediapipe.solutions import drawing_utils as draw_module
        mpHands = hands_module
        draw = draw_module
        MEDIAPIPE_OK = True
except Exception as e:
    print(f"[WARNING] MediaPipe import failed: {e}")
    print("[INFO] Will use fallback mode (hand detection disabled for now)")
    MEDIAPIPE_OK = False
    USE_MEDIAPIPE = False

if MEDIAPIPE_OK and mpHands:
    try:
        hands = mpHands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7
        )
    except Exception as e:
        print(f"[WARNING] Could not initialize Hands: {e}")
        MEDIAPIPE_OK = False
        hands = None

# ===== OPTIONAL: Gemini + Speech Recognition =====
# Read API key from common environment variable names. Do NOT hardcode the key.
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_API_KEY")
# If no env var set, you can hardcode it here for testing (NOT recommended for production)
if not GENAI_API_KEY:
    GENAI_API_KEY = ""  # Your API key
# Allow overriding the model name via env for easy testing (e.g. GENAI_MODEL=models/gemini-2.5-flash)
GENAI_MODEL = os.environ.get("GENAI_MODEL")
genai = None
GEMINI_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False


# ===== CAMERA CONFIG =====
CAMERA_SOURCE = 0  # Change to video file path (string) for pre-recorded video, or DroidCam URL
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
current_mode = "CUBE"
auto_rotate = False
is_fullscreen = False

mouse = Controller()
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
TRACKPAD_W = 0.85
TRACKPAD_H = 0.85
cursor_buf = deque(maxlen=8)
left_clicked = False
right_clicked = False
double_clicked = False
dragging = False

# mouse click-cooldown state (was referenced but never defined before)
last_click_time = 0.0
CLICK_COOLDOWN = 0.4  # seconds

shape_pos = [320, 240]
shape_scale = 40.0
rot_x, rot_y, rot_z = 0.0, 0.0, 0.0

rot_x_buf = deque(maxlen=6)
rot_y_buf = deque(maxlen=6)
scale_buf = deque(maxlen=6)
move_buf_x = deque(maxlen=6)
move_buf_y = deque(maxlen=6)

prev_two_hand_dist = None
prev_center1 = None
prev_center2 = None

last_ai_command = ""
last_ai_status = ""
BUTTONS = {}
command_queue = queue.Queue()

# ===== Push-to-talk state =====
PTT_KEY = kb.Key.ctrl_r          # hold RIGHT CTRL to talk
push_to_talk_held = False        # true while PTT_KEY is physically down
ptt_trigger_event = threading.Event()
voice_ui_state = "idle"          # "idle" | "listening" | "processing"
last_voice_status = ""           # last recognized phrase / error, shown briefly in the UI

# ===== On-screen keyboard text input state =====
ui_input_mode = False            # true while the on-screen text box is capturing keystrokes
ui_input_text = ""
ui_input_status = ""

# ===== Air-draw state (was referenced but never initialized before) =====
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR: Blue, Green, Red, Yellow
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = green_index = red_index = yellow_index = 0
colorIndex = 0
drawing_filters = {}
drawing_active = False

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

# ===== Pre-built "Quick Load" 3D Models (instant, no API call - for presentations/demos) =====

def create_benzene_ring(n=6, carbon_r=0.8, hydrogen_r=1.3, depth=0.22):
    """Benzene (C6H6) skeletal ring, given real depth (front+back layers) so it reads as a
    3D ring from any angle instead of a flat 2D hexagon."""
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    cf = [[carbon_r*math.cos(a), carbon_r*math.sin(a), -depth/2] for a in angles]
    cb = [[p[0], p[1], depth/2] for p in cf]
    hf = [[hydrogen_r*math.cos(a), hydrogen_r*math.sin(a), -depth/2] for a in angles]
    hb = [[p[0], p[1], depth/2] for p in hf]
    vertices = np.array(cf + cb + hf + hb, dtype=np.float32)

    edges = []
    for i in range(n):
        j = (i+1) % n
        edges.append((i, j))                       # front carbon ring
        edges.append((n+i, n+j))                    # back carbon ring
        edges.append((i, n+i))                       # front-to-back carbon strut
        edges.append((i, 2*n+i))                     # front C-H bond
        edges.append((n+i, 3*n+i))                   # back C-H bond
    return vertices, edges


def create_tetrahedral_carbon():
    """Central carbon with 4 bonds in tetrahedral geometry - ball-and-stick style
    (central atom only, spokes out to the 4 outer atoms - no cage between the outer atoms,
    matching a real methane/tetrahedral-carbon diagram)."""
    raw = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], dtype=np.float32)
    raw = (raw / np.linalg.norm(raw[0])) * 1.6  # bond length tuned to match the other presets' visual size
    vertices = np.vstack([[[0,0,0]], raw]).astype(np.float32)
    edges = [(0,1),(0,2),(0,3),(0,4)]   # only the 4 bonds from the central atom
    return vertices, edges


def create_dna_helix(n_turns=3, points_per_turn=32, radius=0.45, rung_every=5):
    """Double helix: two rotated sine/cosine strands with periodic base-pair rungs."""
    n = n_turns * points_per_turn
    t = np.linspace(0, n_turns*2*math.pi, n)
    y = np.linspace(1, -1, n)

    strand1 = np.stack([radius*np.cos(t), y, radius*np.sin(t)], axis=1)
    strand2 = np.stack([radius*np.cos(t+math.pi), y, radius*np.sin(t+math.pi)], axis=1)
    vertices = np.vstack([strand1, strand2]).astype(np.float32)

    edges = []
    for i in range(n-1):
        edges.append((i, i+1))
        edges.append((n+i, n+i+1))
    for i in range(0, n, rung_every):
        edges.append((i, n+i))
    return vertices, edges


def create_water_molecule():
    """H2O ball-and-stick: central O, two H at the real ~104.5deg bond angle."""
    angle = math.radians(104.5)
    bond_len = 1.1
    half = angle / 2
    o = [0, 0, 0]
    h1 = [bond_len*math.sin(half), bond_len*math.cos(half), 0]
    h2 = [-bond_len*math.sin(half), bond_len*math.cos(half), 0]
    vertices = np.array([o, h1, h2], dtype=np.float32)
    edges = [(0, 1), (0, 2)]
    return vertices, edges


def create_ammonia_molecule():
    """NH3 trigonal pyramidal: central N above a triangle of 3 H atoms."""
    n_pos = [0, 0.55, 0]
    radius = 0.95
    depth = -0.35
    h_positions = []
    for i in range(3):
        a = math.radians(90) + i*2*math.pi/3
        h_positions.append([radius*math.cos(a), depth, radius*math.sin(a)])
    vertices = np.array([n_pos] + h_positions, dtype=np.float32)
    edges = [(0, 1), (0, 2), (0, 3)]
    return vertices, edges


def create_nacl_lattice(n=1):
    """Simple cubic ionic lattice (NaCl-type): ions at integer grid points, bonded only
    to their axis-aligned nearest neighbors - the classic ionic-crystal teaching cage."""
    pts = []
    idx = {}
    r = range(-n, n+1)
    for i in r:
        for j in r:
            for k in r:
                idx[(i, j, k)] = len(pts)
                pts.append([i, j, k])
    edges = []
    for (i, j, k), id0 in idx.items():
        for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            neighbor = (i+d[0], j+d[1], k+d[2])
            if neighbor in idx:
                edges.append((id0, idx[neighbor]))
    vertices = np.array(pts, dtype=np.float32)
    maxabs = np.max(np.abs(vertices)) if len(vertices) else 1.0
    vertices = vertices / maxabs
    return vertices.astype(np.float32), edges


def create_diamond_lattice(n=1):
    """Diamond cubic crystal structure: the standard 8-atom tetrahedral basis repeated
    over an n x n x n supercell, with bonds drawn at the tetrahedral nearest-neighbor
    distance - shows carbon's tetrahedral bonding (contrast with flat Graphene)."""
    basis = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]
    ], dtype=np.float32)
    pts = []
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                for b in basis:
                    pts.append(b + np.array([i, j, k]))
    pts = np.array(pts, dtype=np.float32)
    pts = np.unique(np.round(pts, 4), axis=0)

    nn_dist = math.sqrt(3)/4
    edges = []
    for a_i in range(len(pts)):
        for b_i in range(a_i+1, len(pts)):
            d = np.linalg.norm(pts[a_i]-pts[b_i])
            if abs(d - nn_dist) < 0.02:
                edges.append((a_i, b_i))

    center = pts.mean(axis=0)
    v = pts - center
    maxabs = np.max(np.abs(v)) if len(v) else 1.0
    v = v / maxabs
    return v.astype(np.float32), edges


def create_graphene_sheet(n=2, bond=0.5):
    """Flat honeycomb (hexagonal) carbon lattice - a single graphene layer, laid flat in
    the xz-plane so its 2D hexagonal bonding pattern reads clearly from any 3D rotation."""
    a1 = np.array([1.5, math.sqrt(3)/2]) * bond
    a2 = np.array([1.5, -math.sqrt(3)/2]) * bond
    basis = [np.array([0.0, 0.0]), np.array([bond, 0.0])]

    pts = []
    seen = set()
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            origin = i*a1 + j*a2
            for b in basis:
                p = origin + b
                key = (round(float(p[0]), 3), round(float(p[1]), 3))
                if key not in seen:
                    seen.add(key)
                    pts.append(p)
    pts2d = np.array(pts, dtype=np.float32)

    edges = []
    for a_i in range(len(pts2d)):
        for b_i in range(a_i+1, len(pts2d)):
            d = np.linalg.norm(pts2d[a_i]-pts2d[b_i])
            if abs(d - bond) < 0.03:
                edges.append((a_i, b_i))

    verts = np.zeros((len(pts2d), 3), dtype=np.float32)
    verts[:, 0] = pts2d[:, 0]
    verts[:, 2] = pts2d[:, 1]
    verts -= verts.mean(axis=0)
    maxabs = np.max(np.abs(verts)) if len(verts) else 1.0
    verts /= maxabs
    return verts.astype(np.float32), edges


def create_bohr_atom(shells=3):
    """Simplified Bohr/planetary atom model: a small clustered nucleus plus concentric
    electron-orbit rings, each tilted at a different angle for a readable 3D 'atom' look
    (the vertex dots on each ring already render as the electrons)."""
    verts = []
    edges = []

    nucleus_pts = [[0, 0, 0], [0.06, 0.04, 0.02], [-0.05, 0.05, -0.03], [0.03, -0.06, 0.04]]
    verts.extend(nucleus_pts)
    for i in range(1, len(nucleus_pts)):
        edges.append((0, i))

    n_per_ring = 28
    radii = np.linspace(0.5, 1.0, shells)
    for s in range(shells):
        r = radii[s]
        tilt = s * 0.9
        ring_start = len(verts)
        for k in range(n_per_ring):
            a = 2*math.pi*k/n_per_ring
            x = r*math.cos(a)
            y = r*math.sin(a)*math.sin(tilt)
            z = r*math.sin(a)*math.cos(tilt)
            verts.append([x, y, z])
        for k in range(n_per_ring):
            edges.append((ring_start+k, ring_start+(k+1) % n_per_ring))

    return np.array(verts, dtype=np.float32), edges


def _trig_wave_with_axis(func, n=80):
    """Shared builder for the Sin(x)/Cos(x) presets: plots func over [-2pi, 2pi] as a
    flat 2D curve (z=0), plus x/y reference axes through the origin."""
    x = np.linspace(-2*math.pi, 2*math.pi, n)
    x_norm = x / (2*math.pi)
    y = func(x)

    curve_verts = np.stack([x_norm, y, np.zeros(n)], axis=1).astype(np.float32)
    edges = [(i, i+1) for i in range(n-1)]

    off = len(curve_verts)
    axis_verts = [[-1.15, 0, 0], [1.15, 0, 0], [0, -1.25, 0], [0, 1.25, 0]]
    axis_edges = [(off, off+1), (off+2, off+3)]

    verts = np.vstack([curve_verts, np.array(axis_verts, dtype=np.float32)])
    return verts.astype(np.float32), edges + axis_edges


def create_sine_wave_axis(n=80):
    return _trig_wave_with_axis(np.sin, n)


def create_cosine_wave_axis(n=80):
    return _trig_wave_with_axis(np.cos, n)


def create_parabola_axis(n=60):
    """y = x^2 with x/y reference axes - a flat 2D quadratic-graph teaching diagram."""
    x = np.linspace(-1, 1, n)
    y = x**2
    curve_verts = np.stack([x, y, np.zeros(n)], axis=1).astype(np.float32)
    edges = [(i, i+1) for i in range(n-1)]

    off = len(curve_verts)
    axis_verts = [[-1.15, -0.15, 0], [1.15, -0.15, 0], [0, -0.15, 0], [0, 1.15, 0]]
    axis_edges = [(off, off+1), (off+2, off+3)]
    verts = np.vstack([curve_verts, np.array(axis_verts, dtype=np.float32)])
    return verts.astype(np.float32), edges + axis_edges


def create_unit_circle_trig(n=48, theta_deg=40):
    """Unit circle with x/y axes plus a radius line and its sin/cos projection - a flat
    2D trigonometry 'unit circle' diagram."""
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    circle_verts = np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1).astype(np.float32)
    edges = [(i, (i+1) % n) for i in range(n)]

    off = len(circle_verts)
    theta = math.radians(theta_deg)
    cx, cy = math.cos(theta), math.sin(theta)
    extra_verts = [
        [-1.3, 0, 0], [1.3, 0, 0],
        [0, -1.3, 0], [0, 1.3, 0],
        [0, 0, 0], [cx, cy, 0],
        [cx, 0, 0],
    ]
    extra_edges = [
        (off, off+1), (off+2, off+3),
        (off+4, off+5),
        (off+5, off+6), (off+6, off+4),
    ]
    verts = np.vstack([circle_verts, np.array(extra_verts, dtype=np.float32)])
    return verts.astype(np.float32), edges + extra_edges


def create_solenoid_coil(turns=8, points_per_turn=16, radius=0.55):
    """Tightly-wound single-strand helix plus a straight core axis - a solenoid /
    electromagnet coil diagram for physics."""
    n = turns * points_per_turn
    t = np.linspace(0, turns*2*math.pi, n)
    y = np.linspace(1, -1, n)
    coil = np.stack([radius*np.cos(t), y, radius*np.sin(t)], axis=1).astype(np.float32)
    edges = [(i, i+1) for i in range(n-1)]
    off = len(coil)
    axis_pts = np.array([[0, 1.15, 0], [0, -1.15, 0]], dtype=np.float32)
    verts = np.vstack([coil, axis_pts])
    edges.append((off, off+1))
    return verts.astype(np.float32), edges


def create_optics_prism():
    """Triangular prism plus an incident ray and two dispersed exit rays - the classic
    light-dispersion optics diagram."""
    prism_v, prism_e = create_polygon_prism(3, extrude_depth=0.7)
    verts = list(prism_v)
    edges = list(prism_e)
    off = len(verts)
    incident = [[-1.6, 0.9, 0], [-0.55, 0.15, 0]]
    ray1 = [[0.55, -0.1, 0], [1.6, -0.55, 0]]
    ray2 = [[0.55, -0.1, 0], [1.6, -0.15, 0]]
    verts += incident + ray1 + ray2
    edges += [(off, off+1), (off+2, off+3), (off+4, off+5)]
    return np.array(verts, dtype=np.float32), edges


benzene_vertices_base, benzene_edges_base = create_benzene_ring()
tetra_carbon_vertices_base, tetra_carbon_edges_base = create_tetrahedral_carbon()
dna_vertices_base, dna_edges_base = create_dna_helix()
water_vertices_base, water_edges_base = create_water_molecule()
ammonia_vertices_base, ammonia_edges_base = create_ammonia_molecule()
nacl_vertices_base, nacl_edges_base = create_nacl_lattice()
diamond_vertices_base, diamond_edges_base = create_diamond_lattice()
graphene_vertices_base, graphene_edges_base = create_graphene_sheet()
atom_vertices_base, atom_edges_base = create_bohr_atom()
sinx_vertices_base, sinx_edges_base = create_sine_wave_axis()
cosx_vertices_base, cosx_edges_base = create_cosine_wave_axis()
parabola_vertices_base, parabola_edges_base = create_parabola_axis()
unitcircle_vertices_base, unitcircle_edges_base = create_unit_circle_trig()
solenoid_vertices_base, solenoid_edges_base = create_solenoid_coil()
prism_vertices_base, prism_edges_base = create_optics_prism()

# Preset 3D models - basic geometric primitives merged together with the class 10-12
# exhibition set (chemistry, physics, and math) into a single catalog. This is the one
# list used by both AI mode and 3D MODELS mode, so each shape only needs to be defined
# once and is reachable, by name, from either mode - see announce_models_mode().
PRESET_MODELS = {
    "Cube": (cube_vertices_base, cube_edges_base),
    "Pyramid": (pyramid_vertices_base, pyramid_edges_base),
    "Sphere": (sphere_vertices_base, sphere_edges_base),
    "Triangle": (triangle_vertices_base, triangle_edges_base),
    "Pentagon": (pentagon_vertices_base, pentagon_edges_base),
    "Hexagon": (hexagon_vertices_base, hexagon_edges_base),
    "Octagon": (octagon_vertices_base, octagon_edges_base),
    "Rhombus": (rhombus_vertices_base, rhombus_edges_base),
    "Benzene": (benzene_vertices_base, benzene_edges_base),
    "Graphene": (graphene_vertices_base, graphene_edges_base),
    "Diamond": (diamond_vertices_base, diamond_edges_base),
    "NaCl Lattice": (nacl_vertices_base, nacl_edges_base),
    "Tetrahedral": (tetra_carbon_vertices_base, tetra_carbon_edges_base),
    "Water": (water_vertices_base, water_edges_base),
    "Ammonia": (ammonia_vertices_base, ammonia_edges_base),
    "DNA": (dna_vertices_base, dna_edges_base),
    "Atom": (atom_vertices_base, atom_edges_base),
    "Solenoid": (solenoid_vertices_base, solenoid_edges_base),
    "Prism": (prism_vertices_base, prism_edges_base),
    "Sin(x)": (sinx_vertices_base, sinx_edges_base),
    "Cos(x)": (cosx_vertices_base, cosx_edges_base),
    "Parabola": (parabola_vertices_base, parabola_edges_base),
    "Unit Circle": (unitcircle_vertices_base, unitcircle_edges_base),
}

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
hand_filters = {}

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

# ===== Drawing Shape =====
WIREFRAME_LINE_THICKNESS = 4
WIREFRAME_LINE_COLOR = (60, 255, 90)
WIREFRAME_JOINT_RADIUS = 6
WIREFRAME_JOINT_COLOR = (0, 165, 255)      # filled joint dot (orange - pops against the green lines)
WIREFRAME_JOINT_RING_COLOR = (255, 255, 255)  # thin white outline so joints read clearly on any background

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

    # Thick, anti-aliased wireframe edges
    for e in shape_edges:
        a,b = e
        if a < 0 or b < 0 or a >= len(v) or b >= len(v): continue
        cv2.line(frame, tuple(v[a][:2]), tuple(v[b][:2]), WIREFRAME_LINE_COLOR,
                  WIREFRAME_LINE_THICKNESS, cv2.LINE_AA)

    # Joint dots at every vertex so the point-cloud structure of the model is visible
    for pt in v:
        p = tuple(pt[:2])
        cv2.circle(frame, p, WIREFRAME_JOINT_RADIUS, WIREFRAME_JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, p, WIREFRAME_JOINT_RADIUS, WIREFRAME_JOINT_RING_COLOR, 1, cv2.LINE_AA)

# ====== AR GESTURES ======
def handle_ar_gestures(frame, results):
    global shape_pos, shape_scale, rot_x, rot_y, prev_two_hand_dist
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

        if idx1 not in hand_filters:
            hand_filters[idx1] = KalmanFilter()
        if idx2 not in hand_filters:
            hand_filters[idx2] = KalmanFilter()

        c1 = hand_filters[idx1].update(lm1[9])
        c2 = hand_filters[idx2].update(lm2[9])

        mid_x = int((c1[0] + c2[0]) / 2)
        mid_y = int((c1[1] + c2[1]) / 2)
        dist = np.linalg.norm(c1 - c2)

        open1 = sum(f1) >= 3
        open2 = sum(f2) >= 3
        index1 = (f1 == [0,1,0,0,0])
        index2 = (f2 == [0,1,0,0,0])

        if open1 and open2:
            if prev_two_hand_dist is not None:
                diff = dist - prev_two_hand_dist
                new_scale = shape_scale + diff * 1.0
                new_scale = np.clip(new_scale, 10, 600)
                scale_buf.append(new_scale)
                shape_scale = sum(scale_buf)/len(scale_buf) if scale_buf else new_scale
            prev_two_hand_dist = dist

            dx = (mid_x - w/2) / (w/2)
            dy = (mid_y - h/2) / (h/2)
            rot_y_buf.append(dx * math.pi * 1.2)
            rot_x_buf.append(dy * math.pi * 1.2)
            rot_y = sum(rot_y_buf)/len(rot_y_buf)
            rot_x = sum(rot_x_buf)/len(rot_x_buf)

        elif index1 and index2:
            move_buf_x.append(mid_x)
            move_buf_y.append(mid_y)
            shape_pos[0] = sum(move_buf_x)/len(move_buf_x)
            shape_pos[1] = sum(move_buf_y)/len(move_buf_y)

        else:
            prev_two_hand_dist = dist

    else:
        (lm, f, _) = hands_info[0]
        if f == [0,1,0,0,0]:
            move_buf_x.append(lm[9][0])
            move_buf_y.append(lm[9][1])
            shape_pos[0] = sum(move_buf_x)/len(move_buf_x)
            shape_pos[1] = sum(move_buf_y)/len(move_buf_y)

# ===== Mouse Mode =====
def smooth_cursor(x, y, w, h):
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

    measured = (sx, sy)
    pred = mouse_filter.update(measured)

    cursor_buf.append(pred)
    if len(cursor_buf) >= 4:
        avg_x = sum(p[0] for p in cursor_buf) / len(cursor_buf)
        avg_y = sum(p[1] for p in cursor_buf) / len(cursor_buf)
    else:
        avg_x, avg_y = pred

    avg_y = min(max(avg_y, 0), screen_h)
    avg_x = min(max(avg_x, 0), screen_w)

    mouse.position = (avg_x, avg_y)

def detect_mouse_gesture(frame,lm,w,h,is_left):
    global left_clicked, right_clicked, double_clicked, dragging, last_click_time

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

    # Pinch gestures are edge-triggered so holding a pinch cannot repeat clicks.
    # Index+middle pinch = double-click, middle pinch = right-click, index pinch = left-click.
    if pinch_idx and pinch_mid:
        if not double_clicked:
            mouse.click(Button.left)
            time.sleep(0.12)
            mouse.click(Button.left)
            double_clicked = True
        left_clicked = True
        right_clicked = True
    elif pinch_mid:
        if not right_clicked:
            mouse.click(Button.right)
            right_clicked = True
        left_clicked = False
        double_clicked = False
    elif pinch_idx:
        if not left_clicked:
            mouse.click(Button.left)
            left_clicked = True
        right_clicked = False
        double_clicked = False
    else:
        left_clicked = False
        right_clicked = False
        double_clicked = False

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

# ===== Letter/Number Template Generation (3D EXTRUSION) =====
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

def match_preset_model(text):
    """Map free-text/voice input to one of the pre-built PRESET_MODELS keys, if it matches.
    Basic geometric primitives (cube, pyramid, ...) are merged into the same preset
    catalog as the chemistry/physics/math models, so there's a single lookup path
    shared by AI mode and 3D MODELS mode."""
    t = text.lower()
    if "cube" in t:
        return "Cube"
    if "pyramid" in t:
        return "Pyramid"
    if "sphere" in t or "ball" in t:
        return "Sphere"
    if "octagon" in t:
        return "Octagon"
    if "hexagon" in t:
        return "Hexagon"
    if "pentagon" in t:
        return "Pentagon"
    if "triangle" in t:
        return "Triangle"
    if "rhombus" in t:
        return "Rhombus"
    if "benzene" in t:
        return "Benzene"
    if "graphene" in t:
        return "Graphene"
    if "diamond" in t:
        return "Diamond"
    if "sodium chloride" in t or "nacl" in t or ("salt" in t and "lattice" in t) or "ionic" in t:
        return "NaCl Lattice"
    if "tetrahedral" in t or "methane" in t:
        return "Tetrahedral"
    if "ammonia" in t or "nh3" in t:
        return "Ammonia"
    if "water" in t or "h2o" in t:
        return "Water"
    if "dna" in t:
        return "DNA"
    if "bohr" in t or "atom" in t:
        return "Atom"
    if "solenoid" in t or "coil" in t:
        return "Solenoid"
    if "prism" in t:
        return "Prism"
    if "cosine" in t or re.search(r'\bcos\b', t):
        return "Cos(x)"
    if "sine" in t or ("sign" in t and "wave" in t) or re.search(r'\bsin\b', t):
        return "Sin(x)"
    if "parabola" in t:
        return "Parabola"
    if "unit circle" in t or "trig circle" in t or ("circle" in t and "unit" in t):
        return "Unit Circle"
    return None


def announce_models_mode():
    """3D Models mode has no on-screen list (kept off the camera view to save UI
    space) - print the available names to the terminal instead."""
    try:
        print("\n[3D MODELS] Say or type one of these:")
        print("  " + ", ".join(PRESET_MODELS.keys()))
    except:
        pass


def load_preset_model(key):
    """Instantly load a pre-built model - no API call, so it's presentation-safe."""
    global shape_vertices, shape_edges, current_shape_name, last_ai_status, current_mode
    if key not in PRESET_MODELS:
        return False
    v, e = PRESET_MODELS[key]
    shape_vertices = v.copy()
    shape_edges = list(e)
    current_shape_name = key
    last_ai_status = "Preset model (instant load)"
    current_mode = "MODELS"
    try:
        print(f"[3D MODELS] Loaded: {key}")
    except:
        pass
    return True


def generate_shape_from_text(text):
    global shape_vertices, shape_edges, current_shape_name, last_ai_status
    global last_ai_command, current_mode, genai, GEMINI_AVAILABLE, GENAI_API_KEY

    last_ai_command = text.strip()
    t = last_ai_command.strip().lower()

    preset_key = match_preset_model(t)
    if preset_key:
        load_preset_model(preset_key)
        return

    if current_mode == "MODELS":
        # The pre-built 3D Models section NEVER calls the Gemini API - if the spoken/typed
        # name isn't one of the presets, just report that instead of falling through.
        print(f"[3D MODELS] '{t}' didn't match a preset. Available: " + ", ".join(PRESET_MODELS.keys()))
        last_ai_status = "Not a pre-built model - see terminal"
        return

    letter_match = re.search(r'\b(?:letter|alphabet)\s+([a-z])\b', t)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in letter_shapes:
            shape_vertices, shape_edges = letter_shapes[letter]
            current_shape_name = f"Letter {letter}"
            last_ai_status = "Template letter"
            current_mode = "CUBE"
            return

    # support multi-digit numbers (e.g. 'number 65') as well as word numbers
    number_match = re.search(r'\b(?:number|digit)\s+((?:\d+)|zero|one|two|three|four|five|six|seven|eight|nine)\b', t)
    if number_match:
        num_str = number_match.group(1)
        word_to_digit = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
                         "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        digit = word_to_digit.get(num_str, num_str)
        # If we have a builtin template for the whole token (single digit word/number)
        if digit in number_shapes:
            shape_vertices, shape_edges = number_shapes[digit]
            current_shape_name = f"Number {digit}"
            last_ai_status = "Template number"
            current_mode = "CUBE"
            return

        # If the token is a multi-digit number like '65', build a composite from digit templates
        if digit.isdigit() and len(digit) > 1:
            parts = []
            edges = []
            offset = 0.0
            spacing = 1.6  # spacing multiplier between digit templates
            vertex_offset = 0
            any_missing = False
            # compute a scale to make composite fit nicely
            for ch in digit:
                if ch not in number_shapes:
                    any_missing = True
                    break
            if not any_missing:
                # normalize each digit size and place them side-by-side
                for i,ch in enumerate(digit):
                    v,e = number_shapes[ch]
                    # normalize digit to width ~1.0
                    minx = v[:,0].min()
                    maxx = v[:,0].max()
                    w = maxx - minx if (maxx - minx) != 0 else 1.0
                    scale = 1.0 / w
                    v_scaled = v.copy()
                    v_scaled[:, :2] = (v_scaled[:, :2] - np.array([minx, v_scaled[:,1].mean()])) * scale
                    # translate by offset in x
                    v_scaled[:,0] += offset
                    parts.append(v_scaled)
                    for (a,b) in e:
                        edges.append((vertex_offset + a, vertex_offset + b))
                    vertex_offset += v_scaled.shape[0]
                    offset += spacing
                if parts:
                    shape_vertices = np.vstack(parts).astype(np.float32)
                    shape_edges = edges
                    current_shape_name = f"Number {digit}"
                    last_ai_status = "Template composite number"
                    current_mode = "CUBE"
                    return

    if len(t) == 1:
        if t.isdigit() and t in number_shapes:
            shape_vertices, shape_edges = number_shapes[t]
            current_shape_name = f"Number {t}"
            last_ai_status = "Template number"
            current_mode = "CUBE"
            return
        k = t.upper()
        if k in letter_shapes:
            shape_vertices, shape_edges = letter_shapes[k]
            current_shape_name = f"Letter {k}"
            last_ai_status = "Template letter"
            current_mode = "CUBE"
            return

    word_to_digit = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
                     "five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
    if t in word_to_digit:
        d = word_to_digit[t]
        if d in number_shapes:
            shape_vertices, shape_edges = number_shapes[d]
            current_shape_name = f"Number {d}"
            last_ai_status = "Template (word)"
            current_mode = "CUBE"
            return

    # Basic geometric shapes (cube, pyramid, sphere, ...) are now part of PRESET_MODELS
    # and already handled by the match_preset_model() check above - no separate lookup
    # needed here anymore.

    # Try to initialize Gemini client now if a key exists but it wasn't imported at startup
    if not GEMINI_AVAILABLE:
        if GENAI_API_KEY and genai is None:
            try:
                import google.generativeai as _genai
                _genai.configure(api_key=GENAI_API_KEY)
                genai = _genai
                GEMINI_AVAILABLE = True
            except Exception:
                GEMINI_AVAILABLE = False

        if not GEMINI_AVAILABLE:
            shape_vertices = cube_vertices_base.copy()
            shape_edges = cube_edges_base.copy()
            current_shape_name = "Cube"
            last_ai_status = "Gemini OFF (no API key)"
            current_mode = "CUBE"
            return

    try:
        prompt = f"""You are a 3D wireframe shape generator.
Given the description: "{t}"

Return ONLY a JSON object (no markdown fences, no explanation) in exactly this format:
{{"vertices": [[x,y,z], [x,y,z], ...], "edges": [[i,j], [i,j], ...]}}

Rules:
- Use between 6 and 30 vertices.
- Coordinates should roughly range from -1 to 1 on each axis.
- "edges" is a list of [i, j] pairs of 0-based vertex indices to connect with a line.
- The shape should visually resemble: "{t}".
- Respond with ONLY the JSON object."""
        # Allow explicit override via env var, otherwise try a short list of likely model names
        model_names = []
        if GENAI_MODEL:
            model_names.append(GENAI_MODEL)
        # include models discovered from REST listing (common stable names)
        model_names += ["models/gemini-2.5-flash", "gemini-2.0-flash", "gemini-2", "models/gemini-2", "gemini-1"]
        resp = None
        last_model_exception = None
        for mname in model_names:
            try:
                print(f"Trying generative model: {mname}")
                model = genai.GenerativeModel(mname)
                resp = model.generate_content(prompt)
                break
            except Exception as ex:
                last_model_exception = ex
        if resp is None:
            # Surface helpful message about model/key mismatch with fuller context
            msg = f"Model init/generation failed. Last exception: {last_model_exception}"
            print(msg)
            raise RuntimeError(msg)
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
        last_ai_status = "AI generated"
        current_mode = "CUBE"
    except Exception as e:
        last_ai_status = f"AI error: {str(e)[:30]}"
        shape_vertices = pyramid_vertices_base.copy()
        shape_edges = pyramid_edges_base.copy()
        current_shape_name = "Pyramid"
        current_mode = "CUBE"

def list_available_models():
    """Try to list available Generative AI models using the Generative Language REST endpoint.
    This uses the API key style access as a quick check. It may require OAuth depending on your project.
    """
    key = GENAI_API_KEY
    if not key:
        print("No GENAI API key found in environment variables: GEMINI_API_KEY/GENAI_API_KEY/GOOGLE_GENERATIVE_API_KEY")
        return None
    url = f"https://generativelanguage.googleapis.com/v1/models?key={key}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
            print("Available models (raw):")
            print(json.dumps(data, indent=2))
            print("\nTip: set environment variable GENAI_MODEL to one of the returned 'name' values, e.g.:\n")
            if "models" in data:
                for m in data.get("models", []):
                    print(" - ", m.get("name"))
            return data
    except Exception as ex:
        print(f"Failed to list models via REST endpoint: {ex}")
        print("If this fails, check Generative AI Models in Google Cloud Console or use OAuth credentials.")
        return None


def _on_ptt_key_press(key):
    global push_to_talk_held
    if key == PTT_KEY:
        push_to_talk_held = True
        ptt_trigger_event.set()

def _on_ptt_key_release(key):
    global push_to_talk_held
    if key == PTT_KEY:
        push_to_talk_held = False

def start_ptt_listener():
    """Global push-to-talk key hook (same mechanism as the existing pynput-based mouse
    control, so it works even if the camera window doesn't have OS focus)."""
    try:
        listener = kb.Listener(on_press=_on_ptt_key_press, on_release=_on_ptt_key_release)
        listener.daemon = True
        listener.start()
    except Exception as e:
        try:
            print(f"[PUSH-TO-TALK] Could not start global key listener: {e}")
        except:
            pass

def voice_listener():
    """Push-to-talk: hold PTT_KEY (Right Ctrl) to record one phrase, then release. The
    recognizer's own pause-detection decides exactly when the phrase ends, so you don't
    need to hold the key for the whole sentence - just long enough to start it.
    voice_ui_state / last_voice_status are read every frame by the UI so LISTENING and
    PROCESSING are always shown on screen instead of the app looking frozen."""
    global voice_ui_state, last_voice_status
    if not SR_AVAILABLE: return
    try:
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.9        # a bit more slack so multi-word commands aren't cut off
        r.non_speaking_duration = 0.4

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1.0)
            # A noisy calibration moment can leave the threshold too high (nothing
            # triggers) or too low (everything triggers) - keep it in a sane range.
            r.energy_threshold = min(max(r.energy_threshold, 150), 4000)
            try:
                print(f"[VOICE] Microphone ready (energy threshold={int(r.energy_threshold)}). "
                      f"Hold RIGHT CTRL to talk in AI/3D MODELS mode.")
            except:
                pass

            while True:
                ptt_trigger_event.wait()
                ptt_trigger_event.clear()
                if not push_to_talk_held:
                    continue
                if current_mode not in ("AI", "MODELS"):
                    try:
                        print("[VOICE] Push-to-talk pressed, but not in AI/3D MODELS mode - ignored.")
                    except:
                        pass
                    continue

                voice_ui_state = "listening"
                last_voice_status = ""
                try:
                    audio = r.listen(source, timeout=5, phrase_time_limit=8)
                except sr.WaitTimeoutError:
                    voice_ui_state = "idle"
                    last_voice_status = "No speech detected - try again"
                    continue
                except Exception as e:
                    voice_ui_state = "idle"
                    last_voice_status = f"Mic error: {str(e)[:40]}"
                    continue

                voice_ui_state = "processing"
                try:
                    text = r.recognize_google(audio, language="en-US").lower()
                    try:
                        print(f"[VOICE] Recognized: {text}")
                    except:
                        pass
                    if current_mode in ("AI", "MODELS"):
                        command_queue.put(text)
                    last_voice_status = f"Heard: {text}"
                except sr.UnknownValueError:
                    last_voice_status = "Could not understand - try again or type"
                except sr.RequestError as e:
                    last_voice_status = f"Speech API error: {str(e)[:40]}"
                finally:
                    voice_ui_state = "idle"
    except Exception as e:
        try:
            print(f"[VOICE] Failed to initialize: {e}")
        except:
            pass

def terminal_input_listener():
    """Listen for terminal input (type shape commands) as fallback for voice recognition"""
    try:
        print("\n[TERMINAL INPUT] Type shape commands (e.g., 'cube', 'letter A', 'number 5', 'benzene')")
        print("[TERMINAL INPUT] Press Enter to send. Type 'help' for examples.")
        print("[TERMINAL INPUT] Tip: you can also press T in the camera window to type there,")
        print("[TERMINAL INPUT] or hold RIGHT CTRL to talk (push-to-talk).\n")
    except:
        pass  # Ignore any print errors in daemon thread
    
    import sys
    while True:
        try:
            user_input = input("[SHAPE CMD] ").strip().lower()
            if not user_input:
                continue
            if user_input == "help":
                try:
                    print("Templates:    letter A, number 5")
                    print("Preset shapes & models: " + ", ".join(PRESET_MODELS.keys()))
                except:
                    pass
                continue
            if current_mode in ("AI", "MODELS"):
                try:
                    print(f"[TERMINAL] Sending: {user_input}")
                except:
                    pass
                command_queue.put(user_input)
        except EOFError:
            break
        except KeyboardInterrupt:
            try:
                print("\n[TERMINAL INPUT] Paused (will resume)")
            except:
                pass
        except Exception as e:
            try:
                print(f"[TERMINAL INPUT] Error: {e}")
            except:
                pass

def clear_drawing():
    """Remove strokes only when the user explicitly requests it."""
    global bpoints,gpoints,rpoints,ypoints
    global blue_index,green_index,red_index,yellow_index,drawing_active
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    blue_index = green_index = red_index = yellow_index = 0
    drawing_filters.clear()
    drawing_active = False

def begin_new_stroke():
    global bpoints,gpoints,rpoints,ypoints
    global blue_index,green_index,red_index,yellow_index
    bpoints.append(deque(maxlen=1024))
    gpoints.append(deque(maxlen=1024))
    rpoints.append(deque(maxlen=1024))
    ypoints.append(deque(maxlen=1024))
    blue_index += 1; green_index += 1; red_index += 1; yellow_index += 1

# ===== UI polish helpers =====
def draw_panel(frame, pt1, pt2, alpha=0.55, color=(15,15,20), border=(0,255,200)):
    """Semi-transparent panel behind text/buttons so they stay readable over the camera feed."""
    x1,y1 = pt1; x2,y2 = pt2
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(frame.shape[1],int(x2)), min(frame.shape[0],int(y2))
    if x2<=x1 or y2<=y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
    if border:
        cv2.rectangle(frame,(x1,y1),(x2,y2),border,1,cv2.LINE_AA)

def draw_text_with_shadow(frame, text, org, font, scale, color, thickness=1):
    x,y = org
    cv2.putText(frame,text,(x+1,y+1),font,scale,(0,0,0),thickness+1,cv2.LINE_AA)
    cv2.putText(frame,text,(x,y),font,scale,color,thickness,cv2.LINE_AA)


def draw_mode_buttons(frame):
    global BUTTONS
    h,w = frame.shape[:2]
    menu_y = h - 60
    # MOUSE mode is intentionally hidden from the on-screen bar to save space and
    # reduce clutter - it's still reachable with the '2' key.
    draw_panel(frame, (5, menu_y-5), (480, h-8), alpha=0.5)
    BUTTONS = {
        "CUBE":   ((10, menu_y), (120, menu_y + 45)),
        "DRAW":   ((130, menu_y), (230, menu_y + 45)),
        "AI":     ((240, menu_y), (320, menu_y + 45)),
        "MODELS": ((330, menu_y), (470, menu_y + 45)),
    }
    for mode,((x1,y1),(x2,y2)) in BUTTONS.items():
        active = (mode==current_mode)
        col = (0,255,140) if active else (150,150,150)
        if active:
            overlay = frame.copy()
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,90,60),-1)
            cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        label = "3D MODELS" if mode == "MODELS" else mode
        scale = 1.25 if mode == "MODELS" else 1.6
        draw_text_with_shadow(frame,label,(x1+8,y1+29),cv2.FONT_HERSHEY_PLAIN,scale,col,1)

    if current_mode != "DRAW":
        return

    cv2.rectangle(frame,(40, 1),(140,65),(0,0,0),-1)
    cv2.putText(frame,"CLEAR",(52,38),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    btns = [(160,255,"BLUE"),(275,370,"GREEN"),(390,485,"RED"),(505,600,"YELLOW")]
    for i,(x1,x2,label) in enumerate(btns):
        cv2.rectangle(frame,(x1,1),(x2,65),color[i],-1)
        cv2.putText(frame,label,(x1+10,38),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)


def draw_voice_ptt_ui(frame):
    """Push-to-talk status banner. Shows an animated LISTENING/PROCESSING indicator
    (pulsing color + spinner) while the mic/recognizer are working, so the UI never
    looks frozen - then leaves the last recognized phrase (or error) up briefly."""
    h, w = frame.shape[:2]
    spinner_chars = ['|', '/', '-', '\\']
    spin = spinner_chars[int(time.time() * 6) % 4]

    if voice_ui_state in ("listening", "processing"):
        label = "LISTENING" if voice_ui_state == "listening" else "PROCESSING"
        pulse = (math.sin(time.time() * 6) + 1) / 2  # 0..1 breathing effect
        base = (0, 60, 255) if voice_ui_state == "listening" else (0, 200, 255)  # BGR: red / amber
        col = tuple(int(c * (0.5 + 0.5 * pulse)) for c in base)

        box_w, box_h = 300, 46
        x1 = w // 2 - box_w // 2
        y1 = 12
        draw_panel(frame, (x1, y1), (x1 + box_w, y1 + box_h), alpha=0.65, color=(15, 15, 20), border=col)
        draw_text_with_shadow(frame, f"[MIC] {label} {spin}", (x1 + 16, y1 + 31),
                               cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2)
    elif last_voice_status:
        box_w, box_h = 300, 34
        x1 = w // 2 - box_w // 2
        y1 = 12
        draw_panel(frame, (x1, y1), (x1 + box_w, y1 + box_h), alpha=0.5, color=(15, 15, 20), border=(0, 255, 140))
        draw_text_with_shadow(frame, last_voice_status[:38], (x1 + 12, y1 + 23),
                               cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 140), 1)


def draw_ui_text_input(frame):
    """Small on-screen keyboard text box - type a shape command directly into the
    camera window (press T to open it) instead of switching to the terminal."""
    h, w = frame.shape[:2]
    box_w = 340
    x1 = w - box_w - 10
    y1 = 5

    if ui_input_mode:
        box_h = 66
        draw_panel(frame, (x1, y1), (x1 + box_w, y1 + box_h), alpha=0.7, color=(15, 15, 20), border=(0, 255, 140))
        draw_text_with_shadow(frame, "TYPE A COMMAND:", (x1 + 10, y1 + 20),
                               cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        shown = (ui_input_text + cursor)[-34:]
        draw_text_with_shadow(frame, shown, (x1 + 10, y1 + 43),
                               cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 0), 1)
        draw_text_with_shadow(frame, "[Enter] Send   [Esc] Cancel", (x1 + 10, y1 + 60),
                               cv2.FONT_HERSHEY_PLAIN, 0.9, (170, 170, 170), 1)
    else:
        show_status = bool(ui_input_status)
        box_h = 26 + (22 if show_status else 0)
        draw_panel(frame, (x1, y1), (x1 + box_w, y1 + box_h), alpha=0.45, color=(15, 15, 20), border=(90, 90, 90))
        draw_text_with_shadow(frame, "[T] Type command   [Hold R-CTRL] Talk", (x1 + 10, y1 + 18),
                               cv2.FONT_HERSHEY_PLAIN, 1.0, (170, 170, 170), 1)
        if show_status:
            draw_text_with_shadow(frame, ui_input_status[:42], (x1 + 10, y1 + 40),
                                   cv2.FONT_HERSHEY_PLAIN, 0.95, (0, 255, 140), 1)


def on_mouse(event,x,y,flags,param):
    global current_mode, colorIndex
    if event==cv2.EVENT_LBUTTONDOWN:
        if current_mode == "DRAW" and y <= 65:
            if 40 <= x <= 140:
                clear_drawing()
            elif 160 <= x <= 255: colorIndex = 0
            elif 275 <= x <= 370: colorIndex = 1
            elif 390 <= x <= 485: colorIndex = 2
            elif 505 <= x <= 600: colorIndex = 3
            return
        for mode,((x1,y1),(x2,y2)) in BUTTONS.items():
            if x1<=x<=x2 and y1<=y<=y2:
                if mode == "MODELS" and current_mode != "MODELS":
                    announce_models_mode()
                current_mode=mode
                break

def handle_air_draw_mode(frame, results, w, h):
    global bpoints,gpoints,rpoints,ypoints
    global blue_index,green_index,red_index,yellow_index,colorIndex,drawing_active

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
            clear_drawing()
        elif 160<=index_tip[0]<=255: colorIndex=0
        elif 275<=index_tip[0]<=370: colorIndex=1
        elif 390<=index_tip[0]<=485: colorIndex=2
        elif 505<=index_tip[0]<=600: colorIndex=3

    elif drawing_mode:
        if not drawing_active:
            begin_new_stroke()
            drawing_active = True
        if colorIndex==0: bpoints[blue_index].appendleft(index_tip)
        elif colorIndex==1: gpoints[green_index].appendleft(index_tip)
        elif colorIndex==2: rpoints[red_index].appendleft(index_tip)
        elif colorIndex==3: ypoints[yellow_index].appendleft(index_tip)
    else:
        drawing_active = False

def draw_saved_strokes(frame):
    """Render the drawing every frame, including while no hand is visible."""
    points=[bpoints,gpoints,rpoints,ypoints]
    for i in range(4):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k] and points[i][j][k-1]:
                    cv2.line(frame,points[i][j][k-1],points[i][j][k],color[i],5)


def open_camera(source):
    """Robust camera opener that tries multiple backends and indices."""
    try_sources = []
    if isinstance(source, str) and not source.isdigit():
        try_sources.append(source)
    else:
        idx = int(source)
        try_sources.append(idx)

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for s in try_sources:
        for b in backends:
            try:
                cap = cv2.VideoCapture(s, b)
            except Exception:
                try:
                    cap = cv2.VideoCapture(s)
                except Exception:
                    cap = None
            if cap is not None and cap.isOpened():
                try:
                    ok, _ = cap.read()
                except Exception:
                    ok = False
                if ok:
                    print(f"Opened camera source={s} backend={b}")
                    return cap
                else:
                    cap.release()

    for i in range(0, 4):
        for b in backends:
            try:
                cap = cv2.VideoCapture(i, b)
            except Exception:
                try:
                    cap = cv2.VideoCapture(i)
                except Exception:
                    cap = None
            if cap is not None and cap.isOpened():
                try:
                    ok, _ = cap.read()
                except Exception:
                    ok = False
                if ok:
                    print(f"Opened camera source={i} backend={b}")
                    return cap
                else:
                    cap.release()

    return None


def main():
    global current_mode, auto_rotate, drawing_active, is_fullscreen
    global ui_input_mode, ui_input_text, ui_input_status

    # If the user wants to quickly list available GenAI models, set LIST_GENAI_MODELS=1
    if os.environ.get("LIST_GENAI_MODELS") == "1":
        list_available_models()
        return

    cap = open_camera(CAMERA_SOURCE)

    # IMPORTANT: check for None BEFORE calling .set() on it
    if cap is None or not cap.isOpened():
        print("Camera not opened. Please check your camera device or try another index.")
        msg = np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(msg, "Camera not opened.", (30,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(msg, "Check camera or use a video file.", (30,260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        cv2.imshow("AR + Mouse + AI Shapes", msg)
        while True:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Some Windows backends (MSMF especially) can silently negotiate a resolution that
    # produces a corrupted frame buffer, which crashes cv2.Mat on the next read(). Verify
    # the forced resolution actually works; if not, reopen at the camera's own default
    # instead of forcing 1280x720.
    try:
        test_ok, test_frame = cap.read()
        if not test_ok or test_frame is None:
            raise RuntimeError("post-resize test read returned no frame")
    except Exception as e:
        print(f"[WARNING] {CAMERA_WIDTH}x{CAMERA_HEIGHT} isn't supported cleanly by this "
              f"camera/backend ({e}). Reopening at the camera's default resolution instead.")
        cap.release()
        cap = open_camera(CAMERA_SOURCE)
        if cap is None or not cap.isOpened():
            print("Camera not opened on retry. Please check your camera device.")
            cv2.destroyAllWindows()
            return

    cv2.namedWindow("AR + Mouse + AI Shapes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AR + Mouse + AI Shapes", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setWindowProperty("AR + Mouse + AI Shapes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("AR + Mouse + AI Shapes", on_mouse)

    if SR_AVAILABLE:
        print("[VOICE] Speech recognition available. Starting push-to-talk listener...")
        threading.Thread(target=voice_listener, daemon=True).start()
        start_ptt_listener()
    else:
        print("[VOICE] Speech recognition NOT available. Using terminal/on-screen typing only.")
    
    # Always start terminal input listener for fallback
    threading.Thread(target=terminal_input_listener, daemon=True).start()
    
    # MediaPipe status
    if not MEDIAPIPE_OK:
        print("\n[ERROR] WARNING: MediaPipe hand tracking is NOT available!")
        print("[INFO] To fix this, try one of these options:")
        print("  Option 1 (Recommended): Downgrade Python to 3.11")
        print("  Option 2: Install conda and use: conda install -c conda-forge mediapipe")
        print("  Option 3: Use AI mode with terminal input (no hand gestures needed)")
        print("[INFO] Available modes:")
        print("  - AI mode: Rotate shapes by typing commands in terminal")
        print("  - CUBE mode: Static shape rotation")
        print("  - DRAW mode: DISABLED (requires MediaPipe)")
        print("  - MOUSE mode: DISABLED (requires MediaPipe)\n")
    else:
        print("[OK] MediaPipe hand tracking is available!")

    print("\n=== CONTROLS ===")
    print("1 - CUBE mode  |  2 - MOUSE mode (keyboard-only)  |  3 - DRAW mode  |  4 - AI mode  |  5 - 3D MODELS mode")
    print("A - Toggle auto-rotate")
    print("F - Toggle fullscreen (starts fullscreen)")
    print("T - Type a command in the on-screen text box (Enter to send, Esc to cancel)")
    print("Hold RIGHT CTRL - Push-to-talk (speak your command, release when done)")
    print("Q - Quit")
    print("(You can also click the on-screen mode buttons - MOUSE is hidden from the bar to save space)")
    print("\n=== 3D MODELS (mode 5) - instant, NO Gemini/API call, ever - shapes + exhibition set ===")
    print("No on-screen list (kept off the camera view to save space) - say or type a name:")
    print(", ".join(PRESET_MODELS.keys()))
    print("Anything else typed/said in this mode is ignored (ai generation only happens in AI mode).")
    print("\n=== AI mode (mode 4) - letters, numbers, preset models, and Gemini for anything else ===")
    print("LETTERS:  'letter A', 'letter B', 'alphabet C', or just 'A'")
    print("NUMBERS:  'number 5', 'digit 7', 'number five', or just '5'")
    print("CUSTOM:   Any other description -> sent to Gemini (if GEMINI_API_KEY set)")
    print("(All preset names above - basic shapes AND the exhibition set - also work in")
    print(" AI mode and load instantly, no API call.)")
    print("\n=== TERMINAL INPUT FALLBACK (AI / 3D MODELS mode) ===")
    print("If voice recognition fails, type shape commands in the terminal.")
    print("Examples: cube, pyramid, benzene, graphene, diamond, dna, sine wave, unit circle, letter A, number 5")
    print("\n=== GESTURES (CUBE/AI/3D MODELS mode) ===")
    print("Two hands open: Scale & rotate shape")
    print("Two index fingers: Move shape")
    print("One index finger: Move shape\n")

    while True:
        try:
            ok, frame = cap.read()
        except cv2.error as e:
            # A corrupted frame buffer from the backend (seen with MSMF) throws here
            # instead of just returning ok=False - reopen the camera and keep going
            # rather than crashing the whole app.
            print(f"[WARNING] Camera read error ({e}). Reopening camera...")
            cap.release()
            cap = open_camera(CAMERA_SOURCE)
            if cap is None or not cap.isOpened():
                print("Could not recover the camera. Exiting.")
                break
            continue
        if not ok or frame is None:
            break
        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection if MediaPipe is available
        if MEDIAPIPE_OK and hands:
            results = hands.process(rgb)
        else:
            # Create mock results object for fallback
            class MockResults:
                multi_hand_landmarks = None
                multi_handedness = None
            results = MockResults()

        if current_mode in ("AI", "MODELS"):
            while not command_queue.empty():
                generate_shape_from_text(command_queue.get())

        if current_mode in ("CUBE","AI","MODELS"):
            if auto_rotate:
                global rot_y
                rot_y += 0.015
            draw_shape(frame)
            if MEDIAPIPE_OK and results.multi_hand_landmarks:
                handle_ar_gestures(frame, results)
                if draw and mpHands:
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
            cv2.putText(frame, "Index pinch: left | Middle pinch: right | Both: double-click",
                        (pad_left + 10, pad_bottom - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
            if MEDIAPIPE_OK and results.multi_hand_landmarks:
                handle_mouse_mode(frame, results, w, h)
            if not MEDIAPIPE_OK:
                cv2.putText(frame, "MediaPipe not available - hand gestures disabled",
                            (pad_left + 10, pad_bottom - 30),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 1)

        elif current_mode == "DRAW":
            if MEDIAPIPE_OK and results.multi_hand_landmarks:
                handle_air_draw_mode(frame,results,w,h)
            else:
                # A returning hand starts a fresh stroke instead of joining across a gap.
                drawing_active = False
            # Keep every stroke visible until the user presses the CLEAR button.
            draw_saved_strokes(frame)
            cv2.putText(frame, "Index up: draw | Two fingers up: pause | C / CLEAR: erase",
                        (10, 135), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)
            if not MEDIAPIPE_OK:
                cv2.putText(frame, "MediaPipe not available - drawing disabled",
                            (10, 100),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

        info_h = 112 if (last_ai_command or last_ai_status) else 68
        draw_panel(frame, (5,5), (340, info_h), alpha=0.55)
        draw_text_with_shadow(frame, f"Mode: {current_mode}", (15,30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,140), 1)
        draw_text_with_shadow(frame, f"Shape: {current_shape_name}", (15,55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,255), 1)
        if last_ai_command:
            draw_text_with_shadow(frame, f"Voice: {last_ai_command}", (15,80), cv2.FONT_HERSHEY_PLAIN, 1.1, (255,255,0), 1)
        if last_ai_status:
            draw_text_with_shadow(frame, last_ai_status, (15,102), cv2.FONT_HERSHEY_PLAIN, 1.0, (190,190,190), 1)

        if not MEDIAPIPE_OK:
            draw_text_with_shadow(frame, "[WARNING] MediaPipe not available - hand tracking disabled",
                        (10,h-90), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,165,255), 1)

        draw_text_with_shadow(frame, "build: v7", (w-110, h-15), cv2.FONT_HERSHEY_PLAIN, 0.85, (120,120,120), 1)

        draw_mode_buttons(frame)
        draw_voice_ptt_ui(frame)
        draw_ui_text_input(frame)

        cv2.imshow("AR + Mouse + AI Shapes", frame)

        # --- THE KEY FIX: pump the GUI event loop & handle keyboard shortcuts ---
        key = cv2.waitKey(1) & 0xFF

        if ui_input_mode:
            # While the on-screen text box is open, every key is text input, not a
            # mode shortcut - this is what keeps '1'-'5', 'a', 'f', 'q' etc. from
            # firing while you're typing a command.
            if key == 27:  # Esc - cancel
                ui_input_mode = False
                ui_input_text = ""
            elif key in (13, 10):  # Enter - submit
                cmd = ui_input_text.strip()
                ui_input_mode = False
                ui_input_text = ""
                if cmd:
                    if current_mode in ("AI", "MODELS"):
                        try:
                            print(f"[UI INPUT] Sending: {cmd}")
                        except:
                            pass
                        command_queue.put(cmd)
                        ui_input_status = f"Sent: {cmd}"
                    else:
                        ui_input_status = "Switch to AI or 3D MODELS mode to send commands"
            elif key in (8, 127):  # Backspace / Delete
                ui_input_text = ui_input_text[:-1]
            elif 32 <= key <= 126:  # printable ASCII
                if len(ui_input_text) < 40:
                    ui_input_text += chr(key)
            # any other key (arrows, function keys, etc.) is ignored while typing
        else:
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = "CUBE"
            elif key == ord('2'):
                current_mode = "MOUSE"
            elif key == ord('3'):
                current_mode = "DRAW"
            elif key == ord('4'):
                current_mode = "AI"
            elif key == ord('5'):
                entering = (current_mode != "MODELS")
                if entering and current_shape_name not in PRESET_MODELS:
                    load_preset_model("Benzene")
                if entering:
                    announce_models_mode()
                current_mode = "MODELS"
            elif key == ord('a'):
                auto_rotate = not auto_rotate
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty("AR + Mouse + AI Shapes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("AR + Mouse + AI Shapes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("AR + Mouse + AI Shapes", WINDOW_WIDTH, WINDOW_HEIGHT)
            elif key == ord('c') and current_mode == "DRAW":
                clear_drawing()
            elif key == ord('t'):
                ui_input_mode = True
                ui_input_text = ""
                ui_input_status = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
