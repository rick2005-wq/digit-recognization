import streamlit as st
import numpy as np
import cv2
import time
import json
import random as _random
from PIL import Image
import plotly.graph_objects as go
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model, Model as KModel

# ==============================
# THEME CONSTANTS
# ==============================

BG_DARK       = "#0a0a0f"
BG_PANEL      = "#111118"
ACCENT_CYAN   = "#00e5ff"
ACCENT_PURPLE = "#7c3aed"
ACCENT_PINK   = "#f000b8"
TEXT_PRIMARY  = "#e2e8f0"
TEXT_DIM      = "#4a5568"
CANVAS_BG     = "#050508"
GRID_COLOR    = "#0d1117"

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Neural Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# HTML HELPER
# ==============================

def render_html(text):
    return '\n'.join(line.strip() for line in text.strip().split('\n'))

# ==============================
# CUSTOM CSS — FUTURISTIC UPGRADE
# ==============================

st.markdown(render_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

html, body, .stApp {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Courier New', monospace;
}}
.stApp {{
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 59px, {GRID_COLOR} 59px, {GRID_COLOR} 60px),
        repeating-linear-gradient(90deg, transparent, transparent 59px, {GRID_COLOR} 59px, {GRID_COLOR} 60px);
    background-attachment: fixed;
}}
.particle {{
    position: fixed; border-radius: 50%; pointer-events: none; z-index: 0;
    animation: floatUp linear infinite;
}}
@keyframes floatUp {{
    0% {{ transform: translateY(100vh); opacity: 0; }}
    10% {{ opacity: 0.6; }} 90% {{ opacity: 0.6; }}
    100% {{ transform: translateY(-10vh) translateX(20px); opacity: 0; }}
}}
.corner-glow {{
    position: fixed; width: 200px; height: 200px; border-radius: 50%;
    border: 2px solid; pointer-events: none; z-index: 0;
    animation: pulseGlow 3s ease-in-out infinite;
}}
@keyframes pulseGlow {{
    0%, 100% {{ opacity: 0.1; transform: scale(1); }}
    50% {{ opacity: 0.38; transform: scale(1.1); }}
}}
h1, h2, h3 {{
    font-family: 'Orbitron', 'Courier New', monospace !important;
    color: {ACCENT_CYAN}; letter-spacing: 3px; text-transform: uppercase;
}}
.glitch-title {{
    font-family: 'Orbitron', monospace; font-size: 2.3rem; font-weight: 900;
    color: {ACCENT_CYAN}; letter-spacing: 4px; text-transform: uppercase;
    animation: glitchAnim 7s ease-in-out infinite;
    text-shadow: 0 0 30px rgba(0,229,255,0.35);
}}
@keyframes glitchAnim {{
    0%, 88%, 100% {{ text-shadow: 0 0 30px rgba(0,229,255,0.35); transform: translate(0); }}
    89% {{ text-shadow: 2px 0 {ACCENT_PINK}, -2px 0 {ACCENT_CYAN}; transform: translate(2px,-1px); }}
    90% {{ text-shadow: -2px 0 {ACCENT_PINK}, 2px 0 {ACCENT_CYAN}; transform: translate(-1px,1px); }}
    91% {{ text-shadow: 2px 0 {ACCENT_PINK}, -2px 0 {ACCENT_CYAN}; transform: translate(0); }}
    92% {{ text-shadow: none; transform: translate(1px,0); }}
    93% {{ text-shadow: 0 0 30px rgba(0,229,255,0.35); transform: translate(0); }}
}}
.blink-cursor::after {{
    content: '█'; animation: blink 1.06s step-end infinite;
    margin-left: 6px; color: {ACCENT_CYAN};
}}
@keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
.scan-line-container {{
    width: 100%; max-width: 520px; height: 3px;
    background: {BG_DARK}; position: relative; overflow: hidden;
    margin: 8px auto 0 auto;
}}
.scan-line-bar {{
    position: absolute; top: 0; left: 0; width: 60px; height: 100%;
    background: {ACCENT_CYAN}; animation: scanMove 2s linear infinite;
}}
@keyframes scanMove {{ 0% {{ left: 0; }} 100% {{ left: 100%; }} }}
/* Glass panels */
.glass-panel {{
    background: rgba(17,17,24,0.82);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0,229,255,0.13); border-radius: 2px;
    padding: 14px 18px; margin-top: 10px; position: relative;
}}
.glass-panel::before {{
    content: ''; position: absolute; top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, {ACCENT_CYAN}, transparent);
}}
/* HUD corner brackets */
.hud-wrap {{ position: relative; display: inline-block; }}
.hud-c {{ position: absolute; width: 14px; height: 14px; border-color: {ACCENT_CYAN}; border-style: solid; z-index: 5; }}
.hud-tl {{ top:-2px; left:-2px; border-width: 2px 0 0 2px; }}
.hud-tr {{ top:-2px; right:-2px; border-width: 2px 2px 0 0; }}
.hud-bl {{ bottom:-2px; left:-2px; border-width: 0 0 2px 2px; }}
.hud-br {{ bottom:-2px; right:-2px; border-width: 0 2px 2px 0; }}
/* Buttons */
.stButton > button {{
    font-family: 'Orbitron', monospace !important; font-weight: 700 !important;
    font-size: 0.7rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; border: 1px solid {ACCENT_CYAN} !important;
    border-radius: 0 !important; background: rgba(0,0,12,0.85) !important;
    color: {ACCENT_CYAN} !important;
    box-shadow: 0 0 8px rgba(0,229,255,0.2), inset 0 0 6px rgba(0,229,255,0.03) !important;
    transition: all 0.25s ease !important;
}}
.stButton > button:hover {{
    background: rgba(0,229,255,0.08) !important;
    box-shadow: 0 0 22px rgba(0,229,255,0.55), inset 0 0 14px rgba(0,229,255,0.08) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}
/* Slider */
div[data-testid="stSlider"] label {{
    color: {TEXT_DIM} !important; font-family: 'Courier New', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}
/* Checkbox */
.stCheckbox label {{
    color: {ACCENT_PURPLE} !important; font-family: 'Courier New', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 1px !important;
}}
/* Result panel */
.result-panel {{
    background: rgba(17,17,24,0.88);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(0,229,255,0.18);
    padding: 14px 20px; margin-top: 1rem; position: relative; overflow: hidden;
}}
.result-panel-active {{
    animation: activeBorderPulse 0.8s ease-out;
}}
@keyframes activeBorderPulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(0,229,255,0.6); }}
    100% {{ box-shadow: 0 0 24px 4px rgba(0,229,255,0); }}
}}
.result-panel::before {{
    content: ''; position: absolute; top:0; left:0; width:100%; height:1px;
    background: linear-gradient(90deg, transparent, {ACCENT_CYAN}, transparent);
}}
/* Scanning overlay */
.scanning-panel::after {{
    content: ''; position: absolute; top:0; left:0; width:100%; height:2px;
    background: {ACCENT_CYAN}; box-shadow: 0 0 10px {ACCENT_CYAN};
    animation: scanDown 0.8s linear forwards; z-index:10;
}}
@keyframes scanDown {{ 0% {{ top:0; opacity:1; }} 100% {{ top:100%; opacity:0; }} }}
/* Digit flash */
@keyframes flashDigit {{
    0%,100% {{ color:{ACCENT_CYAN}; text-shadow:0 0 20px rgba(0,229,255,0.7); }}
    30% {{ color:{TEXT_DIM}; text-shadow:none; }}
    60% {{ color:{ACCENT_CYAN}; text-shadow:0 0 20px rgba(0,229,255,0.7); }}
}}
.flash-digit {{ animation: flashDigit 0.4s step-end 1; }}
/* Confidence bar */
.conf-bar-bg {{
    width:100%; height:10px; background:#1a1a2e;
    position:relative; overflow:hidden; border:1px solid #1a1a2e;
}}
.conf-bar-fill {{
    height:100%;
    background: linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_CYAN});
    transition: width 0.5s ease;
    box-shadow: 0 0 14px rgba(0,229,255,0.5);
}}
/* Right panel cards */
.rp-header {{
    font-family: 'Orbitron', monospace; font-size: 0.65rem;
    color: {ACCENT_CYAN}; letter-spacing: 2px; margin-bottom: 6px;
    text-transform: uppercase; border-bottom: 1px solid rgba(0,229,255,0.15);
    padding-bottom: 4px;
}}
/* Metrics */
div[data-testid="metric-container"] {{
    background: rgba(17,17,24,0.8) !important;
    border: 1px solid rgba(0,229,255,0.12) !important;
    border-radius: 2px !important; padding: 8px 12px !important;
}}
div[data-testid="metric-container"] label {{
    color: {TEXT_DIM} !important; font-family: 'Courier New', monospace !important;
    font-size: 0.65rem !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {ACCENT_CYAN} !important; font-family: 'Orbitron', monospace !important;
    font-size: 1.3rem !important;
}}
/* Canvas area */
.canvas-wrapper {{
    position:relative; display:inline-block;
    border: 2px solid {ACCENT_CYAN}; padding:2px;
    background:{ACCENT_PURPLE};
    box-shadow: 0 0 20px rgba(0,229,255,0.2);
}}
.canvas-grid-overlay {{
    position:absolute; top:0; left:0; width:100%; height:100%;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 33px, #0f1520 33px, #0f1520 34px),
        repeating-linear-gradient(90deg, transparent, transparent 33px, #0f1520 33px, #0f1520 34px);
    pointer-events:none; z-index:1;
}}
.canvas-corner {{
    position:absolute; font-family:'Courier New',monospace;
    font-size:0.65rem; color:{TEXT_DIM}; z-index:2; pointer-events:none;
}}
/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_DARK}; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT_PURPLE}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_CYAN}; }}
/* Footer */
.footer {{
    text-align:center; color:{TEXT_DIM}; font-size:0.75rem;
    letter-spacing:2px; font-family:'Courier New',monospace;
    margin-top:2rem; padding-bottom:1.5rem;
}}
/* Expander */
.streamlit-expanderHeader {{
    font-family:'Orbitron',monospace !important; color:{ACCENT_CYAN} !important;
    font-size:0.7rem !important; letter-spacing:2px !important;
    background: rgba(17,17,24,0.8) !important;
}}
</style>
"""), unsafe_allow_html=True)

# ==============================
# ANIMATED BACKGROUND
# ==============================

particles_html = ""
for i in range(22):
    px    = _random.randint(0, 100)
    size  = _random.choice([2, 3, 4])
    color = _random.choice([ACCENT_CYAN, ACCENT_PURPLE, ACCENT_PINK, "#005566", "#3a0060", "#004433"])
    dur   = _random.uniform(9, 22)
    delay = _random.uniform(0, 12)
    particles_html += (
        f'<div class="particle" style="left:{px}vw;bottom:-10px;width:{size}px;'
        f'height:{size}px;background:{color};animation-duration:{dur}s;'
        f'animation-delay:{delay}s;"></div>'
    )

glows_html = (
    f'<div class="corner-glow" style="top:-50px;left:-50px;border-color:{ACCENT_PURPLE};"></div>'
    f'<div class="corner-glow" style="top:-50px;right:-50px;border-color:{ACCENT_CYAN};animation-delay:0.7s"></div>'
    f'<div class="corner-glow" style="bottom:-50px;left:-50px;border-color:{ACCENT_PINK};animation-delay:1.4s"></div>'
    f'<div class="corner-glow" style="bottom:-50px;right:-50px;border-color:{ACCENT_PURPLE};animation-delay:2.1s"></div>'
)
st.markdown(particles_html + glows_html, unsafe_allow_html=True)

# ==============================
# SESSION STATE INIT
# ==============================

if "canvas_size" not in st.session_state:
    st.session_state.canvas_size        = 500
    st.session_state.image              = Image.new("L", (500, 500), color=255)
    st.session_state.prediction_history = []
    st.session_state.last_prediction    = None
    st.session_state.all_digits_data    = []
    st.session_state.avg_pred           = None
    st.session_state.model_loaded       = False
    st.session_state.recognition_count  = 0
    st.session_state.scanning           = False
    st.session_state.flash_key          = 0
    st.session_state.canvas_key         = "canvas_v2"
    st.session_state.live_mode          = False
    st.session_state.prev_canvas_hash   = None
    st.session_state.activation_data    = None
    st.session_state.brush_size         = 20
    st.session_state.total_predictions  = 0
    st.session_state.start_time         = time.time()

# ==============================
# LOAD MODEL + BUILD SUB-MODELS
# ==============================

@st.cache_resource
def load_trained_model():
    try:
        return load_model("model/digit_model.keras")
    except Exception:
        return None

@st.cache_resource
def build_activation_model(_model):
    """One multi-output model → single forward pass for all layer activations."""
    if _model is None:
        return None
    try:
        return KModel(
            inputs=_model.input,
            outputs=[
                _model.layers[0].output,   # conv1  (26,26,32)
                _model.layers[1].output,   # pool1  (13,13,32)
                _model.layers[2].output,   # conv2  (11,11,64)
                _model.layers[3].output,   # pool2  (5,5,64)
                _model.layers[5].output,   # dense  (64,)
            ]
        )
    except Exception:
        return None

model     = load_trained_model()
act_model = build_activation_model(model)

if model:
    st.session_state.model_loaded = True

# ==============================
# ACTIVATION EXTRACTION
# ==============================

def _norm(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros(len(arr))
    return (arr - mn) / (mx - mn)

def extract_activations(img_shaped, probs):
    """Return dict of per-layer activation strengths (0-1) for the NN map."""
    if act_model is None:
        return None
    try:
        outs   = act_model.predict(img_shaped, verbose=0)
        c1     = outs[0][0]   # (26,26,32)
        p1     = outs[1][0]   # (13,13,32)
        c2     = outs[2][0]   # (11,11,64)
        p2     = outs[3][0]   # (5,5,64)
        d      = outs[4][0]   # (64,)

        def sm(arr):
            return arr.mean(axis=(0, 1))   # spatial mean → per filter

        c1_8 = _norm(sm(c1).reshape(8, 4).mean(axis=1)).tolist()
        p1_4 = _norm(sm(p1).reshape(4, 8).mean(axis=1)).tolist()
        c2_8 = _norm(sm(c2).reshape(8, 8).mean(axis=1)).tolist()
        p2_4 = _norm(sm(p2).reshape(4, 16).mean(axis=1)).tolist()
        d_8  = _norm(d.reshape(8, 8).mean(axis=1)).tolist()

        return {
            "input": [1.0],
            "conv1": c1_8,
            "pool1": p1_4,
            "conv2": c2_8,
            "pool2": p2_4,
            "dense": d_8,
            "out":   [float(p) for p in probs],
        }
    except Exception as e:
        print(f"Activation extraction error: {e}")
        return None

# ==============================
# PREPROCESSING
# ==============================

def run_preprocessing_multi():
    img_array = st.session_state.image
    if isinstance(img_array, Image.Image):
        raw = np.array(img_array)
    else:
        raw = img_array
        
    # The canvas uses black background, white strokes (from RGBA conversion)
    # The `raw` is already grayscaled. We WANT inverse threshold (digits white, bg black).
    # In earlier iteration, `cv2.THRESH_BINARY_INV` was used, but the canvas setup uses
    # black background and white strokes. This meant white strokes on black inverted to
    # black strokes on white.
    
    # Let's directly threshold to ensure black background and white digit pixels.
    _, thresh = cv2.threshold(raw, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    results = []
    import scipy.ndimage as ndi
    
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        digit_crop  = thresh[y:y+h, x:x+w]
        
        # In MNIST, digits are fitted into a 20x20 box inside a 28x28 image, preserving aspect ratio.
        # Resize to fit 20x20
        if w > h:
            scale = 20.0 / w
            new_w, new_h = 20, int(h * scale)
        else:
            scale = 20.0 / h
            new_w, new_h = int(w * scale), 20
            
        new_h, new_w = max(1, new_h), max(1, new_w)
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center of mass logic
        base = np.zeros((28, 28), dtype=np.uint8)
        
        # Calculate center of mass of the resized image
        cy, cx = ndi.center_of_mass(resized)
        
        # Offset to center of mass (14, 14)
        shift_x = 14 - int(cx + 0.5)
        shift_y = 14 - int(cy + 0.5)
        
        # Prevent out-of-bounds
        start_y = max(0, shift_y)
        end_y = min(28, shift_y + new_h)
        start_x = max(0, shift_x)
        end_x = min(28, shift_x + new_w)
        
        img_y_start = max(0, -shift_y)
        img_y_end = min(new_h, 28 - shift_y)
        img_x_start = max(0, -shift_x)
        img_x_end = min(new_w, 28 - shift_x)
        
        base[start_y:end_y, start_x:end_x] = resized[img_y_start:img_y_end, img_x_start:img_x_end]
        
        norm   = base / 255.0
        shaped = norm.reshape(1, 28, 28, 1)
        results.append((shaped, norm))
    return results

def run_preprocessing():
    digits = run_preprocessing_multi()
    if not digits:
        return None, None
    return digits[-1][0], digits[-1][1]

# ==============================
# PREDICTION FUNCTIONS
# ==============================

def predict_digits():
    if not st.session_state.model_loaded:
        st.error("Model not loaded!")
        return False
    digits_data = run_preprocessing_multi()
    if not digits_data:
        st.warning("No digits detected. Please draw something!")
        return False
    recognized = []
    for shaped, img28 in digits_data:
        pred = model.predict(shaped, verbose=0)[0]
        recognized.append((pred, shaped, img28))

    final_number    = "".join(str(int(np.argmax(p))) for p, _, __ in recognized)
    avg_pred        = np.mean([p for p, _, __ in recognized], axis=0)
    last_pred, last_shaped, last_img28 = recognized[-1]
    avg_confidence  = float(np.mean([np.max(p) for p, _, __ in recognized])) * 100

    st.session_state.all_digits_data   = recognized
    st.session_state.avg_pred          = avg_pred
    st.session_state.last_prediction   = {
        "number":          final_number,
        "last_digit":      int(np.argmax(last_pred)),
        "last_confidence": float(np.max(last_pred)) * 100,
        "avg_confidence":  avg_confidence,
        "all_predictions": recognized,
        "last_img28":      last_img28,
    }
    st.session_state.recognition_count  = len(recognized)
    st.session_state.total_predictions += 1
    st.session_state.prediction_history.append(final_number)

    # Extract real activations (use last digit's shaped input)
    st.session_state.activation_data = extract_activations(last_shaped, last_pred)

    st.session_state.scanning  = True
    st.session_state.flash_key += 1
    return True


def realtime_predict():
    if not st.session_state.model_loaded:
        return
    digits_data = run_preprocessing_multi()
    if not digits_data:
        return
    recognized = []
    for shaped, img28 in digits_data:
        pred = model.predict(shaped, verbose=0)[0]
        recognized.append((pred, shaped, img28))

    final_number   = "".join(str(int(np.argmax(p))) for p, _, __ in recognized)
    avg_pred       = np.mean([p for p, _, __ in recognized], axis=0)
    last_pred, last_shaped, last_img28 = recognized[-1]
    avg_confidence = float(np.mean([np.max(p) for p, _, __ in recognized])) * 100

    st.session_state.all_digits_data  = recognized
    st.session_state.avg_pred         = avg_pred
    st.session_state.last_prediction  = {
        "number":          final_number,
        "last_digit":      int(np.argmax(last_pred)),
        "last_confidence": float(np.max(last_pred)) * 100,
        "avg_confidence":  avg_confidence,
        "all_predictions": recognized,
        "last_img28":      last_img28,
    }
    st.session_state.recognition_count = len(recognized)
    st.session_state.total_predictions += 1
    st.session_state.prediction_history.append(final_number)
    st.session_state.activation_data   = extract_activations(last_shaped, last_pred)


def clear_all():
    st.session_state.image             = Image.new("L", (500, 500), color=255)
    st.session_state.last_prediction   = None
    st.session_state.avg_pred          = None
    st.session_state.all_digits_data   = []
    st.session_state.recognition_count = 0
    st.session_state.scanning          = False
    st.session_state.activation_data   = None
    st.session_state.prediction_history.clear()
    st.session_state.flash_key        += 1
    st.session_state.canvas_key        = f"canvas_{time.time()}"

# ==============================
# ======  UI LAYOUT  ===========
# ==============================

# --- HEADER ---
st.markdown(render_html(f"""
<div style="text-align:center; margin-top:1.2rem; margin-bottom:0.5rem;">
<div class="glitch-title blink-cursor">NEURAL DIGIT RECOGNITION</div>
<div style="color:{TEXT_DIM}; font-size:0.78rem; letter-spacing:4px;
font-family:'Courier New',monospace; margin-top:6px;">
draw · predict · recognize
</div>
<div class="scan-line-container"><div class="scan-line-bar"></div></div>
</div>
"""), unsafe_allow_html=True)

# --- TOP METRICS STRIP ---
uptime_s = int(time.time() - st.session_state.start_time)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("🧠 PREDICTIONS", st.session_state.total_predictions)
mc2.metric("🎯 DIGITS FOUND", st.session_state.recognition_count)
mc3.metric("📡 MODEL", "CNN MNIST")
mc4.metric("⏱ UPTIME", f"{uptime_s//60}m {uptime_s%60}s")

st.markdown("<hr style='border-color:rgba(0,229,255,0.1);margin:6px 0 12px 0;'>", unsafe_allow_html=True)

# --- MAIN COLUMNS ---
col_left, col_right = st.columns([2.1, 1.1], gap="large")

# ==========================================
#  LEFT COLUMN — Canvas + Controls + Result
# ==========================================
with col_left:
    st.markdown("### 📐 Drawing Canvas")

    # Brush size control
    brush_size = st.slider(
        "BRUSH SIZE",
        min_value=6, max_value=36, value=st.session_state.brush_size,
        step=2, key="brush_slider"
    )
    if brush_size != st.session_state.brush_size:
        st.session_state.brush_size = brush_size

    # Canvas wrapper with HUD corners
    scan_overlay = ""
    if st.session_state.scanning:
        scan_overlay = (
            '<div style="position:absolute;top:0;left:0;width:100%;height:2px;'
            f'background:{ACCENT_CYAN};box-shadow:0 0 10px {ACCENT_CYAN};'
            'animation:scanDown 0.9s linear forwards;z-index:10;"></div>'
        )

    st.markdown(render_html(f"""
    <div class="canvas-wrapper">
    <div style="position:relative; background:{CANVAS_BG};">
    {scan_overlay}
    <div class="canvas-grid-overlay"></div>
    <div class="canvas-corner" style="top:4px;left:6px;">00</div>
    <div class="canvas-corner" style="top:4px;right:6px;">FF</div>
    <div class="canvas-corner" style="bottom:4px;left:6px;">FF</div>
    <div class="canvas-corner" style="bottom:4px;right:6px;">00</div>
    </div>
    </div>
    """), unsafe_allow_html=True)

    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=st.session_state.brush_size,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(5, 5, 8)",
        height=500,
        width=500,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
    )

    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        gray      = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        current_img = Image.fromarray(gray)
        st.session_state.image = current_img
        if st.session_state.live_mode:
            img_hash = hash(current_img.tobytes())
            if img_hash != st.session_state.prev_canvas_hash:
                st.session_state.prev_canvas_hash = img_hash
                realtime_predict()

    st.markdown("<div style='margin:0.8rem 0;'></div>", unsafe_allow_html=True)

    # --- BUTTONS ---
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns([1, 1, 1, 1, 0.8])

    with btn_c1:
        if st.button("▶ PREDICT", use_container_width=True, key="btn_predict"):
            if predict_digits():
                st.session_state.scanning = True
                st.rerun()

    with btn_c2:
        if st.button("✕ CLEAR", use_container_width=True, key="btn_clear"):
            clear_all()
            st.rerun()

    with btn_c3:
        if st.button("⚡ LIVE RUN", use_container_width=True, key="btn_live"):
            if predict_digits():
                st.rerun()

    with btn_c4:
        if st.button("↩ RESET", use_container_width=True, key="btn_reset"):
            clear_all()
            st.rerun()

    with btn_c5:
        live_check = st.checkbox("AUTO", value=st.session_state.live_mode, key="live_check")
        if live_check != st.session_state.live_mode:
            st.session_state.live_mode = live_check
            st.rerun()

    # Keyboard hints
    st.markdown(render_html(f"""
    <div style="color:{TEXT_DIM}; font-size:0.65rem; font-family:'Courier New',monospace;
    letter-spacing:1px; margin-top:4px;">
    TIP: PREDICT after drawing · CLEAR to reset · AUTO for live recognition
    </div>
    """), unsafe_allow_html=True)

    # --- RESULT PANEL ---
    if st.session_state.last_prediction:
        pred        = st.session_state.last_prediction
        scan_class  = "scanning-panel" if st.session_state.scanning else ""
        flash_class = "flash-digit"    if st.session_state.scanning else ""

        st.markdown(render_html(f"""
        <div class="result-panel result-panel-active {scan_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="text-align:center; min-width:110px;">
        <div style="font-size:0.62rem; color:{TEXT_DIM}; letter-spacing:3px;
        text-transform:uppercase; margin-bottom:2px; font-family:'Orbitron',monospace;">RECOGNIZED</div>
        <div class="{flash_class}"
        style="font-size:3.8rem; font-weight:900; color:{ACCENT_CYAN};
        font-family:'Orbitron',monospace; text-shadow:0 0 24px rgba(0,229,255,0.6);">
        {pred['number']}
        </div>
        </div>
        <div style="text-align:right;">
        <div style="font-size:0.75rem; color:{TEXT_DIM}; font-family:'Courier New',monospace;">
        {st.session_state.recognition_count} digit(s) found ✓</div>
        <div style="font-size:0.8rem; color:{ACCENT_CYAN};
        font-family:'Orbitron',monospace; margin-top:4px;">
        {pred['avg_confidence']:.1f}%</div>
        <div style="font-size:0.7rem; color:{ACCENT_PURPLE};
        font-family:'Courier New',monospace; margin-top:4px;">CNN · MNIST</div>
        </div>
        </div>
        <div style="margin-top:10px;">
        <div style="font-size:0.6rem; color:{TEXT_DIM}; letter-spacing:2px;
        text-transform:uppercase; margin-bottom:3px;font-family:'Orbitron',monospace;">CONFIDENCE</div>
        <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{min(pred['avg_confidence'],100):.1f}%;"></div>
        </div>
        <div style="text-align:right; font-size:0.7rem; color:{ACCENT_PINK};
        font-family:'Courier New',monospace; margin-top:2px;">
        {pred['avg_confidence']:.2f}%
        </div>
        </div>
        </div>
        """), unsafe_allow_html=True)

        # Probability bars (Plotly – animated, no matplotlib)
        if st.session_state.avg_pred is not None:
            probs = st.session_state.avg_pred
            best  = int(np.argmax(probs))
            colors_bars = [ACCENT_CYAN if i == best else ACCENT_PURPLE for i in range(10)]

            fig_bars = go.Figure(go.Bar(
                x=list(range(10)),
                y=probs,
                marker=dict(
                    color=colors_bars,
                    line=dict(color='#1a1a2e', width=1),
                ),
                text=[f"{p*100:.0f}%" if p > 0.03 else "" for p in probs],
                textposition="inside",
                textfont=dict(color="white", size=9, family="Courier New"),
                hovertemplate="<b>Digit %{x}</b><br>P: %{y:.3f}<extra></extra>",
                showlegend=False,
            ))
            fig_bars.update_layout(
                plot_bgcolor=BG_DARK, paper_bgcolor=BG_DARK,
                font=dict(family="Courier New", color=TEXT_DIM, size=10),
                height=130, margin=dict(l=0, r=0, t=8, b=28),
                xaxis=dict(showgrid=False, dtick=1, tickfont=dict(color=TEXT_DIM, size=9)),
                yaxis=dict(showgrid=True, gridcolor="#1a1a2e", zeroline=False,
                           range=[0, max(float(probs.max()) * 1.25, 0.05)],
                           tickfont=dict(color=TEXT_DIM, size=8)),
            )
            st.plotly_chart(fig_bars, use_container_width=True, config={"displayModeBar": False})

# ==========================================
#  FULL-WIDTH PREDICTION HISTORY BAR
# ==========================================
_arrow = f'<span style="color:{TEXT_DIM};font-size:0.7rem;margin-right:6px;">&rarr;</span>'
if st.session_state.prediction_history:
    _chips = _arrow.join(
        f'<span style="display:inline-block;margin-right:6px;background:rgba(0,229,255,0.07);'
        f'border:1px solid rgba(0,229,255,0.25);padding:2px 10px;'
        f'font-family:Orbitron,monospace;font-size:1rem;font-weight:700;color:{ACCENT_CYAN};'
        f'border-radius:2px;">{e}</span>'
        for e in st.session_state.prediction_history
    )
else:
    _chips = (f'<span style="color:{TEXT_DIM};font-family:Courier New,monospace;'
              f'font-size:0.8rem;font-style:italic;">'
              f'no predictions yet &mdash; draw a digit and click PREDICT</span>')

st.markdown(
    f'<div style="margin:10px 0 14px 0;background:rgba(10,10,20,0.85);'
    f'border:1px solid rgba(0,229,255,0.18);border-left:3px solid {ACCENT_CYAN};'
    f'padding:8px 16px;position:relative;overflow:hidden;">'
    f'<div style="position:absolute;top:0;left:0;width:100%;height:1px;'
    f'background:linear-gradient(90deg,transparent,{ACCENT_CYAN},transparent);"></div>'
    f'<span style="color:{TEXT_DIM};font-family:Orbitron,monospace;font-size:0.6rem;'
    f'letter-spacing:2px;text-transform:uppercase;display:block;margin-bottom:5px;">'
    f'&#x23F1; PREDICTION HISTORY</span>'
    f'<div style="overflow-x:auto;white-space:nowrap;max-height:40px;padding-bottom:2px;">'
    f'{_chips}</div></div>',
    unsafe_allow_html=True
)

# ==========================================
#  RIGHT COLUMN — Intelligence Hub
# ==========================================
with col_right:
    st.markdown("### 📡 Intelligence Hub")

    # --- CONFIDENCE GRAPH ---
    if st.session_state.last_prediction and st.session_state.avg_pred is not None:
        probs = st.session_state.avg_pred
        best  = int(np.argmax(probs))

        st.markdown('<div class="rp-header">CONFIDENCE GRAPH</div>', unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=list(range(10)), y=probs,
            marker=dict(
                color=[ACCENT_CYAN if i == best else ACCENT_PURPLE for i in range(10)],
                line=dict(color="#1a1a2e", width=1),
            ),
            text=[f"{p*100:.0f}%" for p in probs], textposition="inside",
            textfont=dict(color="white", size=9),
            hovertemplate="<b>Digit %{x}</b><br>P: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor=BG_PANEL, paper_bgcolor=BG_PANEL,
            font=dict(family="Courier New", color=TEXT_PRIMARY, size=10),
            height=190, margin=dict(l=30, r=10, t=10, b=30),
            xaxis=dict(showgrid=True, gridcolor="#1a1a2e", dtick=1, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#1a1a2e", zeroline=False,
                       range=[0, float(probs.max()) * 1.2 + 0.02]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # TOP 3 PREDICTIONS
        st.markdown('<div class="rp-header" style="margin-top:10px;">TOP PREDICTIONS</div>', unsafe_allow_html=True)
        top3 = np.argsort(probs)[::-1][:3]
        rank_syms = ["◈", "◇", "○"]
        for rank, idx in enumerate(top3):
            bar_w = int(probs[idx] * 100)
            st.markdown(render_html(f"""
            <div style="padding:6px 8px; background:rgba(10,10,20,0.7);
            border-left:2px solid {ACCENT_PURPLE if rank > 0 else ACCENT_CYAN};
            margin-bottom:4px; position:relative; overflow:hidden;">
            <div style="position:absolute;top:0;left:0;height:100%;
            width:{bar_w}%;background:rgba(0,229,255,0.04);z-index:0;"></div>
            <div style="position:relative;z-index:1;">
            <span style="color:{TEXT_DIM};font-size:0.65rem;
            font-family:'Courier New',monospace;">{rank_syms[rank]} RANK {rank+1}</span>
            <div style="color:{ACCENT_CYAN if rank==0 else TEXT_PRIMARY};
            font-weight:bold;font-family:'Orbitron',monospace;font-size:0.8rem;
            margin-top:1px;">Digit {idx} &nbsp;·&nbsp; {probs[idx]*100:.1f}%</div>
            </div>
            </div>
            """), unsafe_allow_html=True)

    # (History is shown full-width below the canvas — omitted from right panel)

    # --- CNN INPUT 28×28 PREVIEW ---
    st.markdown('<div class="rp-header" style="margin-top:10px;">CNN INPUT [28×28]</div>', unsafe_allow_html=True)
    if st.session_state.last_prediction:
        last_img28 = st.session_state.last_prediction["last_img28"]
        display    = (last_img28 * 255).astype(np.uint8)
        pil_p      = Image.fromarray(display, mode="L").resize((112, 112), Image.NEAREST)
        st.image(pil_p, use_container_width=False)
    else:
        st.markdown(render_html(f"""
        <div style="width:112px;height:112px;background:{CANVAS_BG};
        border:1px dashed rgba(0,229,255,0.15);display:flex;align-items:center;
        justify-content:center;color:{TEXT_DIM};font-family:'Courier New',monospace;
        font-size:0.65rem;text-align:center;">28×28<br>preview</div>
        """), unsafe_allow_html=True)

    # ---- MODEL ARCHITECTURE EXPANDER ----
    with st.expander("⚙ MODEL ARCHITECTURE"):
        st.markdown(render_html(f"""
        <div style="font-family:'Courier New',monospace;font-size:0.7rem;
        color:{TEXT_DIM};line-height:1.8;">
        <span style="color:{ACCENT_CYAN};">INPUT</span>&nbsp;&nbsp;&nbsp;28×28×1<br>
        <span style="color:{ACCENT_CYAN};">CONV1</span>&nbsp;&nbsp;Conv2D(32, 3×3) → ReLU<br>
        <span style="color:{ACCENT_CYAN};">POOL1</span>&nbsp;&nbsp;MaxPool(2×2)<br>
        <span style="color:{ACCENT_CYAN};">CONV2</span>&nbsp;&nbsp;Conv2D(64, 3×3) → ReLU<br>
        <span style="color:{ACCENT_CYAN};">POOL2</span>&nbsp;&nbsp;MaxPool(2×2)<br>
        <span style="color:{ACCENT_CYAN};">FLAT</span>&nbsp;&nbsp;&nbsp;Flatten → 1600<br>
        <span style="color:{ACCENT_CYAN};">FC</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dense(64) → ReLU<br>
        <span style="color:{ACCENT_CYAN};">DROP</span>&nbsp;&nbsp;&nbsp;Dropout(0.5)<br>
        <span style="color:{ACCENT_CYAN};">OUTPUT</span>&nbsp;Dense(10) → Softmax<br>
        <br><span style="color:{ACCENT_PINK};">~93K parameters · 99%+ accuracy</span>
        </div>
        """), unsafe_allow_html=True)

    # ==========================================
    # NEURAL NETWORK MAP — REAL ACTIVATIONS
    # ==========================================
    st.markdown('<div class="rp-header" style="margin-top:10px;">NEURAL NETWORK MAP</div>', unsafe_allow_html=True)

    # Prepare activation JSON + winner list
    if st.session_state.activation_data is not None:
        act_json   = json.dumps(st.session_state.activation_data)
        is_idle    = "false"
    else:
        act_json = json.dumps({
            "input": [0.0], "conv1": [0.0]*8, "pool1": [0.0]*4,
            "conv2": [0.0]*8, "pool2": [0.0]*4, "dense": [0.0]*8,
            "out":   [0.0]*10,
        })
        is_idle = "true"

    if st.session_state.all_digits_data:
        winners = [int(np.argmax(p)) for p, _, __ in st.session_state.all_digits_data]
    elif st.session_state.avg_pred is not None:
        winners = [int(np.argmax(st.session_state.avg_pred))]
    else:
        winners = []

    nn_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ margin:0; background:{BG_PANEL}; overflow:hidden; }}
svg {{ display:block; }}
#tooltip {{
  position:absolute; display:none; background:rgba(0,0,12,0.9);
  border:1px solid {ACCENT_CYAN}; color:{ACCENT_CYAN};
  font-family:monospace; font-size:10px; padding:4px 8px;
  pointer-events:none; z-index:100; white-space:nowrap;
}}
</style>
</head>
<body>
<div id="tooltip"></div>
<svg id="nn-svg" width="400" height="520" style="background:{BG_PANEL};"></svg>
<script>
const ACTS    = {act_json};
const WINNERS = {json.dumps(winners)};
const IS_IDLE = {is_idle};
const CYAN    = '{ACCENT_CYAN}';
const PURPLE  = '{ACCENT_PURPLE}';
const PINK    = '{ACCENT_PINK}';
const DIM     = '{TEXT_DIM}';
const PANEL   = '{BG_PANEL}';

const NN_W = 400, NN_H = 520;
const LAYER_X     = [26, 80, 134, 192, 246, 306, 374];
const LAYER_SIZES = [1,  8,  4,   8,   4,   8,  10];
const LAYER_NAMES = ["INPUT","CONV1","POOL1","CONV2","POOL2","FC","OUTPUT"];
const LAYER_KEYS  = ["input","conv1","pool1","conv2","pool2","dense","out"];
const NODE_R = 5;
const TOP_PAD = 22;
const BOT_PAD = 12;

const svg     = document.getElementById('nn-svg');
const tooltip = document.getElementById('tooltip');

// SVG filters
const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
defs.innerHTML = `
  <filter id="glow">
    <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="sglow">
    <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="xglow">
    <feGaussianBlur in="SourceGraphic" stdDeviation="10" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>`;
svg.appendChild(defs);

// Positions
function getPos(layerIdx) {{
  const n = LAYER_SIZES[layerIdx];
  const space = (NN_H - TOP_PAD - BOT_PAD) / (n + 1);
  return Array.from({{length: n}}, (_, i) => ({{
    x: LAYER_X[layerIdx],
    y: TOP_PAD + space * (i + 1)
  }}));
}}
const allPos = LAYER_X.map((_, l) => getPos(l));

// Color ramp: 0→dark, 0.5→purple, 1→cyan
function actColor(v) {{
  v = Math.max(0, Math.min(1, v));
  if (v < 0.5) {{
    const t = v * 2;
    return [Math.round(15 + 109*t), Math.round(15 + 43*t), Math.round(30 + 207*t)];
  }} else {{
    const t = (v - 0.5) * 2;
    return [Math.round(124 - 124*t), Math.round(58 + 171*t), Math.round(237 + 18*t)];
  }}
}}
function rgb(c) {{ return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})` }}
function mkEl(tag, attrs) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}}

// Layer separator lines + labels
for (let l = 0; l < LAYER_X.length; l++) {{
  if (l > 0) {{
    const midX = (LAYER_X[l-1] + LAYER_X[l]) / 2;
    svg.appendChild(mkEl('line', {{
      x1: midX, y1: TOP_PAD, x2: midX, y2: NN_H - BOT_PAD,
      stroke: 'rgba(255,255,255,0.03)', 'stroke-width': 1
    }}));
  }}
  const t = mkEl('text', {{
    x: LAYER_X[l], y: 14, 'text-anchor': 'middle',
    fill: DIM, 'font-size': 7, 'font-family': 'monospace'
  }});
  t.textContent = LAYER_NAMES[l];
  svg.appendChild(t);
}}

// Edges
const edgeMap = [];
for (let l = 0; l < allPos.length - 1; l++) {{
  const lEdges = [];
  for (let i = 0; i < allPos[l].length; i++) {{
    const iEdges = [];
    for (let j = 0; j < allPos[l+1].length; j++) {{
      const ln = mkEl('line', {{
        x1: allPos[l][i].x,   y1: allPos[l][i].y,
        x2: allPos[l+1][j].x, y2: allPos[l+1][j].y,
        stroke: '#1a1a2e', 'stroke-width': 0.6, opacity: 0.25
      }});
      svg.appendChild(ln);
      iEdges.push(ln);
    }}
    lEdges.push(iEdges);
  }}
  edgeMap.push(lEdges);
}}

// Nodes
const nodeEls = [];
for (let l = 0; l < allPos.length; l++) {{
  const layer = [];
  for (let i = 0; i < allPos[l].length; i++) {{
    const {{x, y}} = allPos[l][i];
    const halo = mkEl('circle', {{cx:x, cy:y, r:NODE_R+5, fill:'none', opacity:0}});
    svg.appendChild(halo);
    const circ = mkEl('circle', {{
      cx:x, cy:y, r:NODE_R, fill:'#1a1a2e', stroke:DIM, 'stroke-width':1
    }});
    svg.appendChild(circ);
    circ.style.cursor = 'pointer';
    const layerKey = LAYER_KEYS[l];
    const actVal   = IS_IDLE ? 0 : (ACTS[layerKey][i] || 0);
    circ.addEventListener('mouseenter', (e) => {{
      tooltip.style.display = 'block';
      tooltip.textContent   = `${{LAYER_NAMES[l]}}[${{i}}]  act=${{actVal.toFixed(3)}}`;
    }});
    circ.addEventListener('mousemove', (e) => {{
      tooltip.style.left = (e.clientX + 10) + 'px';
      tooltip.style.top  = (e.clientY - 24) + 'px';
    }});
    circ.addEventListener('mouseleave', () => {{tooltip.style.display='none';}});
    layer.push({{circ, halo}});
  }}
  nodeEls.push(layer);
}}

// ── Particle with brightness ──
function sendParticle(x1,y1,x2,y2,col,intensity,size) {{
  size = size || 2.5;
  const p = mkEl('circle', {{r: size, fill:col, opacity: 0.5 + intensity*0.5}});
  svg.appendChild(p);
  let t0 = null;
  const dur = 280 + Math.random()*180;
  function step(ts) {{
    if (!t0) t0 = ts;
    const prog = Math.min((ts - t0)/dur, 1);
    p.setAttribute('cx', x1 + (x2-x1)*prog);
    p.setAttribute('cy', y1 + (y2-y1)*prog);
    if (prog < 1) requestAnimationFrame(step);
    else p.remove();
  }}
  requestAnimationFrame(step);
}}

// ── Pulse a node outline ──
function pulseNode(el, color, dur) {{
  dur = dur || 500;
  let t0 = null;
  function step(ts) {{
    if (!t0) t0 = ts;
    const prog = (ts - t0)/dur;
    if (prog >= 1) {{ el.setAttribute('stroke-width',1); return; }}
    el.setAttribute('stroke-width', 1 + Math.sin(prog*Math.PI)*3);
    el.setAttribute('stroke', color);
    requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}}

// ── Per-layer animation ──
function animateLayer(l) {{
  const acts = IS_IDLE
    ? new Array(LAYER_SIZES[l]).fill(0)
    : (ACTS[LAYER_KEYS[l]] || []);

  const isOutput = (l === LAYER_SIZES.length - 1);

  for (let i = 0; i < nodeEls[l].length; i++) {{
    const {{circ, halo}} = nodeEls[l][i];
    const act = acts[i] || 0;
    const col = actColor(act);
    const colS = rgb(col);

    if (isOutput) {{
      if (WINNERS.includes(i)) {{
        // ── WINNER node: blazing bright ──
        circ.setAttribute('fill', CYAN);
        circ.setAttribute('stroke', 'white');
        circ.setAttribute('stroke-width', 2.5);
        circ.setAttribute('r', NODE_R + 2);          // slightly bigger
        circ.setAttribute('filter','url(#xglow)');   // strongest glow

        // Outer ring animated
        halo.setAttribute('stroke', CYAN);
        halo.setAttribute('stroke-width', '1.5');
        halo.setAttribute('fill', 'none');
        halo.setAttribute('opacity', 0.5);
        let rt = 0;
        (function pulseRing() {{
          rt += 0.05;
          halo.setAttribute('r', NODE_R + 6 + Math.sin(rt)*4);
          halo.setAttribute('opacity', 0.3 + Math.sin(rt)*0.2);
          if (document.body.contains(halo)) requestAnimationFrame(pulseRing);
        }})();

        // Second outer ring (offset phase)
        const halo2 = mkEl('circle', {{
          cx: allPos[l][i].x, cy: allPos[l][i].y,
          r: NODE_R+14, stroke: CYAN, 'stroke-width':0.8, fill:'none', opacity:0.15
        }});
        svg.appendChild(halo2);
        let rt2 = Math.PI;
        (function pulseRing2() {{
          rt2 += 0.05;
          halo2.setAttribute('r', NODE_R + 14 + Math.sin(rt2)*4);
          halo2.setAttribute('opacity', 0.08 + Math.sin(rt2)*0.07);
          if (document.body.contains(halo2)) requestAnimationFrame(pulseRing2);
        }})();

        // Big glowing digit label in the node
        const bigLbl = mkEl('text', {{
          x: allPos[l][i].x, y: allPos[l][i].y + 4,
          'text-anchor':'middle', fill: '#000010',
          'font-size': 8, 'font-family': 'monospace', 'font-weight': 'bold'
        }});
        bigLbl.textContent = String(i);
        svg.appendChild(bigLbl);

        // External label below
        const extLbl = mkEl('text', {{
          x: allPos[l][i].x, y: allPos[l][i].y + NODE_R + 14,
          'text-anchor':'middle', fill: CYAN,
          'font-size': 9, 'font-family': 'monospace', 'font-weight': 'bold',
          'filter': 'url(#glow)'
        }});
        extLbl.textContent = String(i);
        svg.appendChild(extLbl);

      }} else {{
        // ── NON-winner outputs: kept very dark ──
        const p = IS_IDLE ? 0 : (ACTS.out ? ACTS.out[i] : 0);
        circ.setAttribute('fill', '#0d0d18');
        circ.setAttribute('stroke', '#1a1a2e');
        circ.setAttribute('stroke-width', 0.5);
        circ.setAttribute('opacity', 0.35 + p * 0.4);
        // Tiny dim digit label
        const lbl = mkEl('text', {{
          x: allPos[l][i].x, y: allPos[l][i].y + NODE_R + 9,
          'text-anchor':'middle', fill: '#2a2a3e', 'font-size':5, 'font-family':'monospace'
        }});
        lbl.textContent = String(i);
        svg.appendChild(lbl);
      }}
    }} else {{
      // ── Hidden layers ──
      circ.setAttribute('fill', IS_IDLE ? '#1a1a2e' : colS);
      circ.setAttribute('stroke', act > 0.55 ? CYAN : (act > 0.3 ? PURPLE : DIM));
      circ.setAttribute('stroke-width', act > 0.65 ? 2 : 1);
      if (act > 0.5 && !IS_IDLE) circ.setAttribute('filter','url(#glow)');
      pulseNode(circ, act > 0.5 ? CYAN : PURPLE, 500);
    }}

    // ── Edges + particles ──
    if (l > 0) {{
      const prevActs = IS_IDLE ? new Array(LAYER_SIZES[l-1]).fill(0)
                               : (ACTS[LAYER_KEYS[l-1]] || []);
      for (let si = 0; si < allPos[l-1].length; si++) {{
        const sa  = prevActs[si] || 0;
        const avg = (sa + act) / 2;
        const eln = edgeMap[l-1][si][i];
        const ec  = actColor(avg);

        // Winner incoming edges glow bright
        const isWinnerEdge = isOutput && WINNERS.includes(i) && !IS_IDLE;
        if (isWinnerEdge) {{
          eln.setAttribute('stroke', CYAN);
          eln.setAttribute('opacity', 0.55);
          eln.setAttribute('stroke-width', 1.2);
        }} else {{
          eln.setAttribute('stroke', IS_IDLE ? '#1a1a2e' : rgb(ec));
          eln.setAttribute('opacity', IS_IDLE ? 0.1 : Math.max(0.04, avg * 0.55));
          eln.setAttribute('stroke-width', 0.6);
        }}

        // Particles: fire along active or winner edges
        if (!IS_IDLE) {{
          const threshold = isWinnerEdge ? 0.0 : 0.3;
          if ((isWinnerEdge || (sa > threshold && act > 0.2)) && Math.random() < (isWinnerEdge ? 0.65 : 0.35)) {{
            const col2 = isWinnerEdge ? CYAN : (sa > 0.7 ? CYAN : PURPLE);
            const sz   = isWinnerEdge ? 3.5 : 2;
            setTimeout(() =>
              sendParticle(allPos[l-1][si].x, allPos[l-1][si].y,
                           allPos[l][i].x, allPos[l][i].y,
                           col2, sa, sz),
              Math.random() * 400
            );
          }}
        }}
      }}
    }}
  }}
}}

// Layer cascade
const LAYER_DELAY = 200;
for (let l = 0; l < LAYER_SIZES.length; l++) {{
  setTimeout(() => animateLayer(l), 60 + l * LAYER_DELAY);
}}

// Continuous particle stream after initial cascade (keeps the map "alive")
if (!IS_IDLE) {{
  setTimeout(() => {{
    setInterval(() => {{
      // Re-fire particles along high-activation paths every 1.2s
      for (let l = 1; l < allPos.length; l++) {{
        const prevActs = ACTS[LAYER_KEYS[l-1]] || [];
        const curActs  = ACTS[LAYER_KEYS[l]]   || [];
        for (let si = 0; si < allPos[l-1].length; si++) {{
          for (let di = 0; di < allPos[l].length; di++) {{
            const sa = prevActs[si] || 0;
            const da = curActs[di] || 0;
            const isWinEdge = (l === allPos.length-1) && WINNERS.includes(di);
            if (isWinEdge || (sa > 0.45 && da > 0.3)) {{
              if (Math.random() < (isWinEdge ? 0.5 : 0.2)) {{
                sendParticle(allPos[l-1][si].x, allPos[l-1][si].y,
                             allPos[l][di].x,   allPos[l][di].y,
                             isWinEdge ? CYAN : PURPLE, sa,
                             isWinEdge ? 3 : 2);
              }}
            }}
          }}
        }}
      }}
    }}, 1200);
  }}, LAYER_SIZES.length * LAYER_DELAY + 500);
}}

// Idle gentle pulse
if (IS_IDLE) {{
  setInterval(() => {{
    for (let l = 0; l < LAYER_SIZES.length; l++) {{
      setTimeout(() => {{
        for (const {{circ}} of nodeEls[l])
          pulseNode(circ, PURPLE, 700);
      }}, l * 120);
    }}
  }}, 4000);
}}
</script>
</body>
</html>"""
    components.html(nn_html, height=535, scrolling=False)

# ==============================
# FOOTER
# ==============================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(render_html(f"""
<div class="footer">
<div style="height:1px;background:linear-gradient(90deg,transparent,{ACCENT_CYAN},transparent);
margin-bottom:10px;"></div>
CNN · MNIST · 99%+ accuracy &nbsp;·&nbsp; Real activation visualization
&nbsp;·&nbsp; Streamlit v2
</div>
"""), unsafe_allow_html=True)

# Reset scan flag (plays only once per prediction)
if st.session_state.scanning:
    st.session_state.scanning = False
