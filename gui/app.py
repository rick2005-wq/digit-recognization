# ==============================
# IMPORT LIBRARIES
# ==============================

import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
import time

from tensorflow.keras.models import load_model


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
GLOW_COLOR    = "#00e5ff"
BTN_PREDICT   = "#00e5ff"
BTN_CLEAR     = "#f000b8"
GRID_COLOR    = "#0d1117"


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = load_model("model/digit_model.keras")


# ==============================
# MAIN WINDOW — fullscreen
# ==============================

root = tk.Tk()
root.title("Neural Digit Recognition")
root.configure(bg=BG_DARK)
root.resizable(True, True)
root.attributes("-fullscreen", True)

root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
root.bind("<F11>",    lambda e: root.attributes("-fullscreen",
                                not root.attributes("-fullscreen")))

root.update_idletasks()
SW = root.winfo_screenwidth()
SH = root.winfo_screenheight()

# Custom font setup
title_font = tkFont.Font(family="Courier", size=13, weight="bold")
label_font = tkFont.Font(family="Courier", size=11)
digit_font = tkFont.Font(family="Courier", size=48, weight="bold")
conf_font  = tkFont.Font(family="Courier", size=11)
btn_font   = tkFont.Font(family="Courier", size=12, weight="bold")
small_font = tkFont.Font(family="Courier", size=9)


# ==============================
# ANIMATED BACKGROUND
# ==============================

import random as _random

bg_canvas = tk.Canvas(root, bg=BG_DARK, highlightthickness=0)
bg_canvas.place(x=0, y=0, relwidth=1.0, relheight=1.0)

# Draw static grid
_GRID_STEP = 60
for _gx in range(0, 3840, _GRID_STEP):
    bg_canvas.create_line(_gx, 0, _gx, 2160, fill="#0d1117", width=1)
for _gy in range(0, 2160, _GRID_STEP):
    bg_canvas.create_line(0, _gy, 3840, _gy, fill="#0d1117", width=1)

# Particles
_NUM_PARTICLES = 55
_particles = []

for _ in range(_NUM_PARTICLES):
    px  = _random.randint(0, SW)
    py  = _random.randint(0, SH)
    spd = _random.uniform(0.15, 0.55)
    r   = _random.choice([1, 1, 1, 2, 2])
    col = _random.choice([
        ACCENT_CYAN, ACCENT_PURPLE, ACCENT_PINK,
        "#005566", "#3a0060", "#004433"
    ])
    oid = bg_canvas.create_oval(px-r, py-r, px+r, py+r, fill=col, outline="")
    _particles.append({"id": oid, "x": px, "y": py, "spd": spd, "r": r,
                        "dx": _random.uniform(-0.3, 0.3)})

def _animate_particles():
    for p in _particles:
        p["y"] -= p["spd"]
        p["x"] += p["dx"]
        if p["y"] < -4:
            p["y"] = SH + 4
            p["x"] = _random.randint(0, SW)
        r = p["r"]
        bg_canvas.coords(p["id"],
                         p["x"]-r, p["y"]-r,
                         p["x"]+r, p["y"]+r)
    root.after(20, _animate_particles)

_animate_particles()

# Slow-pulse corner glows
_GLOW_CORNERS = [
    (0,   0,   200, 200, ACCENT_PURPLE),
    (SW,  0,   SW-200, 200, ACCENT_CYAN),
    (0,   SH,  200, SH-200, ACCENT_PINK),
    (SW,  SH,  SW-200, SH-200, ACCENT_PURPLE),
]
_glow_ids = []
for x1, y1, x2, y2, col in _GLOW_CORNERS:
    gid = bg_canvas.create_oval(
        min(x1,x2), min(y1,y2),
        max(x1,x2), max(y1,y2),
        fill="", outline=col, width=1
    )
    _glow_ids.append(gid)

_glow_alpha = [0]
_glow_dir   = [1]

def _pulse_glows():
    _glow_alpha[0] += _glow_dir[0] * 2
    if _glow_alpha[0] >= 80:
        _glow_dir[0] = -1
    elif _glow_alpha[0] <= 0:
        _glow_dir[0] = 1
    cols = [ACCENT_PURPLE, ACCENT_CYAN, ACCENT_PINK, ACCENT_PURPLE]
    for i, gid in enumerate(_glow_ids):
        a   = _glow_alpha[0]
        hex_col = cols[i]
        r   = int(hex_col[1:3], 16)
        g   = int(hex_col[3:5], 16)
        b   = int(hex_col[5:7], 16)
        dim = max(0, int(r * a / 80))
        dim_g = max(0, int(g * a / 80))
        dim_b = max(0, int(b * a / 80))
        bg_canvas.itemconfig(gid, outline=f"#{dim:02x}{dim_g:02x}{dim_b:02x}")
    root.after(40, _pulse_glows)

_pulse_glows()


# ==============================
# LAYOUT — wider two-column body
# ==============================

# Left col: canvas + existing panels  Right col: confidence graph
LEFT_W  = 560   # fits 500px canvas + padding
RIGHT_W = 280   # new confidence graph panel
GAP     = 20
BODY_W  = LEFT_W + GAP + RIGHT_W   # 860

outer = tk.Frame(root, bg=BG_DARK)
outer.place(relx=0.5, rely=0.0, anchor="n", width=BODY_W, relheight=1.0)
outer.lift()   # raise above background canvas

scroll_c = tk.Canvas(outer, bg=BG_DARK, highlightthickness=0)
scroll_c.pack(fill="both", expand=True)

# Main horizontal container
body = tk.Frame(scroll_c, bg=BG_DARK, width=BODY_W)
scroll_c.create_window((0, 0), window=body, anchor="nw")

body.bind("<Configure>",
          lambda e: scroll_c.configure(scrollregion=scroll_c.bbox("all")))
scroll_c.bind_all("<MouseWheel>",
                  lambda e: scroll_c.yview_scroll(int(-1*(e.delta/120)), "units"))

# Left column (all original content)
col = tk.Frame(body, bg=BG_DARK, width=LEFT_W)
col.grid(row=0, column=0, sticky="n", padx=(0, GAP))
col.grid_propagate(False)

# Right column (confidence graph)
right_col = tk.Frame(body, bg=BG_DARK, width=RIGHT_W)
right_col.grid(row=0, column=1, sticky="n")


# ==============================
# ANIMATED HEADER  (unchanged)
# ==============================

header_frame = tk.Frame(col, bg=BG_DARK)
header_frame.pack(fill="x", padx=30, pady=(24, 0))

title_label = tk.Label(
    header_frame,
    text="NEURAL DIGIT RECOGNITION",
    font=title_font,
    fg=ACCENT_CYAN,
    bg=BG_DARK
)
title_label.pack()

subtitle_label = tk.Label(
    header_frame,
    text="draw  ·  predict  ·  recognize",
    font=small_font,
    fg=TEXT_DIM,
    bg=BG_DARK
)
subtitle_label.pack(pady=(2, 0))

scan_canvas = tk.Canvas(col, width=500, height=3, bg=BG_DARK, highlightthickness=0)
scan_canvas.pack(pady=(8, 0))
scan_line = scan_canvas.create_rectangle(0, 0, 0, 3, fill=ACCENT_CYAN, outline="")

def animate_scan(pos=0, direction=1):
    speed = 6
    pos += speed * direction
    if pos >= 500:
        direction = -1
    elif pos <= 0:
        direction = 1
    scan_canvas.coords(scan_line, pos, 0, pos + 60, 3)
    root.after(12, animate_scan, pos, direction)

animate_scan()


# ==============================
# CANVAS WRAPPER — bigger size
# ==============================

canvas_size = 500

outer_frame = tk.Frame(col, bg=ACCENT_CYAN, padx=2, pady=2)
outer_frame.pack(pady=(16, 0))

inner_frame = tk.Frame(outer_frame, bg=ACCENT_PURPLE, padx=1, pady=1)
inner_frame.pack()

canvas = tk.Canvas(
    inner_frame,
    width=canvas_size,
    height=canvas_size,
    bg=CANVAS_BG,
    cursor="crosshair",
    highlightthickness=0
)
canvas.pack()

def draw_grid():
    step = 34
    for x in range(0, canvas_size, step):
        canvas.create_line(x, 0, x, canvas_size, fill="#0f1520", width=1)
    for y in range(0, canvas_size, step):
        canvas.create_line(0, y, canvas_size, y, fill="#0f1520", width=1)

draw_grid()

canvas.create_text(6, 6,                       text="00", font=small_font, fill=TEXT_DIM, anchor="nw")
canvas.create_text(canvas_size-6, 6,           text="FF", font=small_font, fill=TEXT_DIM, anchor="ne")
canvas.create_text(6, canvas_size-6,           text="FF", font=small_font, fill=TEXT_DIM, anchor="sw")
canvas.create_text(canvas_size-6, canvas_size-6, text="00", font=small_font, fill=TEXT_DIM, anchor="se")

hint_text = canvas.create_text(
    canvas_size // 2, canvas_size // 2,
    text="[ draw here ]",
    font=label_font,
    fill=TEXT_DIM
)

image = Image.new("L", (canvas_size, canvas_size), color=255)
draw_pil = ImageDraw.Draw(image)

is_drawing = False


# ==============================
# DRAWING LOGIC  (unchanged)
# ==============================

def on_press(event):
    global is_drawing
    is_drawing = True
    canvas.delete(hint_text)

def draw_lines(event):
    if not is_drawing:
        return
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r-4, y-r-4, x+r+4, y+r+4,
                       fill="", outline=ACCENT_CYAN, width=1)
    canvas.create_oval(x-r, y-r, x+r, y+r,
                       fill="white", outline="")
    draw_pil.ellipse([x-r, y-r, x+r, y+r], fill=0)

def on_release(event):
    global is_drawing
    is_drawing = False
    # Real-time: trigger prediction 300ms after stroke ends
    root.after(300, realtime_predict)

canvas.bind("<ButtonPress-1>", on_press)
canvas.bind("<B1-Motion>", draw_lines)
canvas.bind("<ButtonRelease-1>", on_release)


# ==============================
# RESULT PANEL  (unchanged)
# ==============================

result_outer = tk.Frame(col, bg=BG_PANEL, padx=2, pady=2)
result_outer.pack(fill="x", padx=30, pady=(18, 0))

result_frame = tk.Frame(result_outer, bg=BG_PANEL, pady=14, padx=20)
result_frame.pack(fill="x")

result_frame.columnconfigure(0, weight=1)
result_frame.columnconfigure(1, weight=0)

digit_var = tk.StringVar(value="?")
digit_display = tk.Label(
    result_frame,
    textvariable=digit_var,
    font=digit_font,
    fg=ACCENT_CYAN,
    bg=BG_PANEL,
    width=3,
    anchor="center"
)
digit_display.grid(row=0, column=0, rowspan=2, sticky="w")

status_var = tk.StringVar(value="awaiting input ...")
status_label = tk.Label(
    result_frame,
    textvariable=status_var,
    font=label_font,
    fg=TEXT_DIM,
    bg=BG_PANEL,
    anchor="e"
)
status_label.grid(row=0, column=1, sticky="e")

conf_var = tk.StringVar(value="CONF:  --.--%")
conf_label = tk.Label(
    result_frame,
    textvariable=conf_var,
    font=conf_font,
    fg=ACCENT_PINK,
    bg=BG_PANEL,
    anchor="e"
)
conf_label.grid(row=1, column=1, sticky="e")


# ==============================
# CONFIDENCE BAR  (unchanged)
# ==============================

bar_frame = tk.Frame(col, bg=BG_DARK)
bar_frame.pack(fill="x", padx=30, pady=(6, 0))

bar_bg = tk.Canvas(bar_frame, width=500, height=10, bg="#1a1a2e",
                   highlightthickness=0)
bar_bg.pack()
conf_bar = bar_bg.create_rectangle(0, 0, 0, 10, fill=ACCENT_CYAN, outline="")


# ==============================
# PROBABILITY BARS  (unchanged)
# ==============================

prob_frame = tk.Frame(col, bg=BG_DARK)
prob_frame.pack(fill="x", padx=30, pady=(10, 0))

prob_bars   = []
prob_labels = []

bar_area = tk.Canvas(prob_frame, width=500, height=56, bg=BG_DARK, highlightthickness=0)
bar_area.pack()

bar_w     = 42
bar_max_h = 36
bar_gap   = 8
start_x   = 14

for i in range(10):
    x0 = start_x + i * (bar_w + bar_gap)
    x1 = x0 + bar_w
    bar_area.create_rectangle(x0, 4, x1, 4 + bar_max_h, fill="#111118", outline=TEXT_DIM)
    bar = bar_area.create_rectangle(x0, 4 + bar_max_h, x1, 4 + bar_max_h,
                                    fill=ACCENT_PURPLE, outline="")
    prob_bars.append(bar)
    lbl = bar_area.create_text(x0 + bar_w//2, 4 + bar_max_h + 12,
                               text=str(i), font=small_font, fill=TEXT_DIM)
    prob_labels.append(lbl)

def update_prob_bars(probs):
    for i, p in enumerate(probs):
        x0 = start_x + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        h  = int(p * bar_max_h)
        color = ACCENT_CYAN if i == np.argmax(probs) else ACCENT_PURPLE
        bar_area.coords(prob_bars[i], x0, 4 + bar_max_h - h, x1, 4 + bar_max_h)
        bar_area.itemconfig(prob_bars[i], fill=color)
        bar_area.itemconfig(prob_labels[i],
                            fill=ACCENT_CYAN if i == np.argmax(probs) else TEXT_DIM)


# ==============================
# BUTTONS  (unchanged)
# ==============================

btn_frame = tk.Frame(col, bg=BG_DARK)
btn_frame.pack(pady=(16, 0))

def make_hover(btn, default_bg, hover_bg, default_fg, hover_fg=BG_DARK):
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg, fg=hover_fg))
    btn.bind("<Leave>", lambda e: btn.config(bg=default_bg, fg=default_fg))

predict_btn = tk.Button(
    btn_frame,
    text="▶  PREDICT",
    font=btn_font,
    bg=BG_PANEL,
    fg=ACCENT_CYAN,
    activebackground=ACCENT_CYAN,
    activeforeground=BG_DARK,
    relief="flat",
    padx=28,
    pady=12,
    bd=0,
    cursor="hand2",
    command=lambda: predict_digit()
)
predict_btn.grid(row=0, column=0, padx=12)
make_hover(predict_btn, BG_PANEL, ACCENT_CYAN, ACCENT_CYAN)

clear_btn = tk.Button(
    btn_frame,
    text="✕  CLEAR",
    font=btn_font,
    bg=BG_PANEL,
    fg=ACCENT_PINK,
    activebackground=ACCENT_PINK,
    activeforeground=BG_DARK,
    relief="flat",
    padx=28,
    pady=12,
    bd=0,
    cursor="hand2",
    command=lambda: clear_canvas()
)
clear_btn.grid(row=0, column=1, padx=12)
make_hover(clear_btn, BG_PANEL, ACCENT_PINK, ACCENT_PINK)


# ==============================
# FOOTER  (unchanged)
# ==============================

footer = tk.Label(
    col,
    text="CNN · MNIST · 99%+ accuracy  ·  Esc to exit fullscreen",
    font=small_font,
    fg=TEXT_DIM,
    bg=BG_DARK
)
footer.pack(pady=(14, 24))


# ==============================
# BLINKING CURSOR  (unchanged)
# ==============================

blink_state = [True]
def blink_cursor():
    if blink_state[0]:
        title_label.config(text="NEURAL DIGIT RECOGNITION █")
    else:
        title_label.config(text="NEURAL DIGIT RECOGNITION  ")
    blink_state[0] = not blink_state[0]
    root.after(530, blink_cursor)

blink_cursor()


# ==============================
# RIGHT PANEL — CONFIDENCE GRAPH
# ==============================

GRAPH_W  = RIGHT_W - 20   # canvas width
GRAPH_H  = 220             # plot area height
GRAPH_PAD_LEFT  = 30      # space for y-axis labels
GRAPH_PAD_BOT   = 20      # space for x-axis labels
GRAPH_PAD_TOP   = 10
GRAPH_PAD_RIGHT = 10

# Header label
rp_header = tk.Label(right_col, text="CONFIDENCE GRAPH",
                     font=small_font, fg=ACCENT_CYAN, bg=BG_DARK)
rp_header.pack(anchor="w", pady=(24, 4))

rp_sep = tk.Canvas(right_col, width=GRAPH_W, height=1,
                   bg=ACCENT_CYAN, highlightthickness=0)
rp_sep.pack(anchor="w", pady=(0, 10))

# Graph canvas
graph_canvas = tk.Canvas(right_col, width=GRAPH_W,
                         height=GRAPH_H + GRAPH_PAD_BOT + GRAPH_PAD_TOP + 4,
                         bg=BG_PANEL, highlightthickness=1,
                         highlightbackground=ACCENT_PURPLE)
graph_canvas.pack(anchor="w")

# Draw static axes
plot_x0 = GRAPH_PAD_LEFT
plot_y0 = GRAPH_PAD_TOP
plot_x1 = GRAPH_W - GRAPH_PAD_RIGHT
plot_y1 = GRAPH_PAD_TOP + GRAPH_H

# Y-axis
graph_canvas.create_line(plot_x0, plot_y0, plot_x0, plot_y1,
                         fill=TEXT_DIM, width=1)
# X-axis
graph_canvas.create_line(plot_x0, plot_y1, plot_x1, plot_y1,
                         fill=TEXT_DIM, width=1)

# Y gridlines + labels (0%, 25%, 50%, 75%, 100%)
for pct in [0, 25, 50, 75, 100]:
    y = plot_y1 - int(pct / 100 * GRAPH_H)
    graph_canvas.create_line(plot_x0, y, plot_x1, y,
                             fill="#1a1a2e", width=1)
    graph_canvas.create_text(plot_x0 - 4, y,
                             text=f"{pct}",
                             font=small_font, fill=TEXT_DIM, anchor="e")

# X-axis digit labels
col_w = (plot_x1 - plot_x0) / 10
for i in range(10):
    cx = plot_x0 + col_w * i + col_w / 2
    graph_canvas.create_text(cx, plot_y1 + 10,
                             text=str(i),
                             font=small_font, fill=TEXT_DIM)

# Bar rectangles for the graph (one per digit)
graph_bars = []
for i in range(10):
    cx   = plot_x0 + col_w * i + col_w / 2
    bw   = col_w * 0.6
    x0b  = cx - bw / 2
    x1b  = cx + bw / 2
    rect = graph_canvas.create_rectangle(
        x0b, plot_y1, x1b, plot_y1,
        fill=ACCENT_PURPLE, outline=""
    )
    graph_bars.append(rect)

# Value label inside top of each bar
graph_val_labels = []
for i in range(10):
    lbl = graph_canvas.create_text(
        0, 0, text="", font=small_font, fill=BG_DARK, anchor="n"
    )
    graph_val_labels.append(lbl)

def update_graph(probs, step=0, target_heights=None):
    best = int(np.argmax(probs))
    if target_heights is None:
        target_heights = [int(p * GRAPH_H) for p in probs]

    MAX_STEPS = 18
    for i in range(10):
        th   = target_heights[i]
        cx   = plot_x0 + col_w * i + col_w / 2
        bw   = col_w * 0.6
        x0b  = cx - bw / 2
        x1b  = cx + bw / 2
        h    = int(th * step / MAX_STEPS)
        color = ACCENT_CYAN if i == best else ACCENT_PURPLE
        graph_canvas.coords(graph_bars[i], x0b, plot_y1 - h, x1b, plot_y1)
        graph_canvas.itemconfig(graph_bars[i], fill=color)

        # Show percentage label inside bar when tall enough
        if h > 14:
            lx = cx
            ly = plot_y1 - h + 4
            graph_canvas.coords(graph_val_labels[i], lx, ly)
            graph_canvas.itemconfig(graph_val_labels[i],
                                    text=f"{int(probs[i]*100)}",
                                    fill=BG_DARK if i == best else TEXT_DIM)
        else:
            graph_canvas.itemconfig(graph_val_labels[i], text="")

    if step < MAX_STEPS:
        root.after(16, update_graph, probs, step + 1, target_heights)

def reset_graph():
    for i in range(10):
        cx   = plot_x0 + col_w * i + col_w / 2
        bw   = col_w * 0.6
        x0b  = cx - bw / 2
        x1b  = cx + bw / 2
        graph_canvas.coords(graph_bars[i], x0b, plot_y1, x1b, plot_y1)
        graph_canvas.itemconfig(graph_bars[i], fill=ACCENT_PURPLE)
        graph_canvas.itemconfig(graph_val_labels[i], text="")

# Top-N label below graph
topn_var = tk.StringVar(value="")
topn_label = tk.Label(right_col, textvariable=topn_var,
                      font=small_font, fg=TEXT_DIM, bg=BG_DARK,
                      justify="left")
topn_label.pack(anchor="w", pady=(8, 0))


# ==============================
# PREDICTION HISTORY
# ==============================

prediction_history = []

# History label in right panel (below topn_label)
history_header = tk.Label(right_col, text="HISTORY",
                          font=small_font, fg=ACCENT_CYAN, bg=BG_DARK)
history_header.pack(anchor="w", pady=(16, 2))

history_label = tk.Label(right_col, text="",
                         font=label_font, fg=ACCENT_PINK, bg=BG_DARK,
                         justify="left")
history_label.pack(anchor="w")


# ==============================
# PROCESSED IMAGE PREVIEW PANEL
# ==============================

preview_header = tk.Label(right_col, text="CNN INPUT  [ 28x28 ]",
                           font=small_font, fg=ACCENT_CYAN, bg=BG_DARK)
preview_header.pack(anchor="w", pady=(16, 2))

PREVIEW_ZOOM = 4
PREVIEW_SIZE = 28 * PREVIEW_ZOOM

preview_outer = tk.Frame(right_col, bg=ACCENT_PURPLE, padx=1, pady=1)
preview_outer.pack(anchor="w")

preview_canvas = tk.Canvas(preview_outer,
                           width=PREVIEW_SIZE, height=PREVIEW_SIZE,
                           bg=CANVAS_BG, highlightthickness=0)
preview_canvas.pack()
preview_img_ref = [None]

def update_preview(img28):
    from PIL import Image as PILImage, ImageTk
    display = (img28 * 255).astype(np.uint8)
    pil_p   = PILImage.fromarray(display, mode="L")
    pil_p   = pil_p.resize((PREVIEW_SIZE, PREVIEW_SIZE), PILImage.NEAREST)
    tk_img  = ImageTk.PhotoImage(pil_p)
    preview_img_ref[0] = tk_img
    preview_canvas.delete("all")
    preview_canvas.create_image(0, 0, anchor="nw", image=tk_img)

def clear_preview():
    preview_canvas.delete("all")
    preview_canvas.create_rectangle(0, 0, PREVIEW_SIZE, PREVIEW_SIZE,
                                    fill=CANVAS_BG, outline="")

clear_preview()




# ==============================
# NEURAL NETWORK VISUALIZER
# ==============================

nn_header = tk.Label(
    right_col,
    text="NEURAL NETWORK MAP",
    font=small_font,
    fg=ACCENT_CYAN,
    bg=BG_DARK
)
nn_header.pack(anchor="w", pady=(18, 4))

NN_W = 250
NN_H = 320

nn_canvas = tk.Canvas(
    right_col,
    width=NN_W,
    height=NN_H,
    bg=BG_PANEL,
    highlightthickness=1,
    highlightbackground=ACCENT_PURPLE
)
nn_canvas.pack(anchor="w")

# Layer x positions and sizes
layer_x     = [30, 80, 130, 180, 230]
layer_sizes = [1,   6,   5,   6,  10]
layer_names = ["IN", "C1", "C2", "D", "OUT"]

network_nodes = []

for layer_idx, size in enumerate(layer_sizes):
    nodes   = []
    spacing = NN_H / (size + 1)
    for i in range(size):
        x = layer_x[layer_idx]
        y = spacing * (i + 1)
        r = 6
        node = nn_canvas.create_oval(
            x-r, y-r, x+r, y+r,
            fill="#1a1a2e",
            outline=TEXT_DIM,
            width=1
        )
        nodes.append(node)
    # Layer label at top
    nn_canvas.create_text(
        layer_x[layer_idx], 6,
        text=layer_names[layer_idx],
        font=small_font,
        fill=TEXT_DIM,
        anchor="n"
    )
    network_nodes.append(nodes)

# Draw connection lines (behind nodes)
network_lines = []
signal_particles = []   # stores (x1,y1,x2,y2) for every inter-layer connection
for l in range(len(network_nodes) - 1):
    layer_lines = []
    for n1 in network_nodes[l]:
        x1, y1, x2, y2 = nn_canvas.coords(n1)
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        for n2 in network_nodes[l + 1]:
            a1, b1, a2, b2 = nn_canvas.coords(n2)
            cx2 = (a1 + a2) / 2
            cy2 = (b1 + b2) / 2
            line = nn_canvas.create_line(
                cx1, cy1, cx2, cy2,
                fill="#1a1a2e", width=1
            )
            layer_lines.append(line)
            signal_particles.append((cx1, cy1, cx2, cy2))
    network_lines.append(layer_lines)

# Raise nodes above lines
for layer in network_nodes:
    for node in layer:
        nn_canvas.tag_raise(node)


# ==============================
# PULSE & SIGNAL-FLOW HELPERS
# ==============================

def pulse_node(node, color, steps=8, step=0):
    """Make a neuron node throb with an expanding outline glow."""
    if step >= steps:
        nn_canvas.itemconfig(node, width=1)
        return
    glow = 1 + (step % 4)
    nn_canvas.itemconfig(node, outline=color, width=glow)
    root.after(40, pulse_node, node, color, steps, step + 1)


def animate_signal(x1, y1, x2, y2, color):
    """Send a glowing particle from (x1,y1) to (x2,y2) along a connection."""
    particle = nn_canvas.create_oval(
        x1 - 3, y1 - 3, x1 + 3, y1 + 3,
        fill=color, outline=""
    )
    STEPS = 20

    def move(step=0):
        if step > STEPS:
            nn_canvas.delete(particle)
            return
        t = step / STEPS
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        nn_canvas.coords(particle, x - 3, y - 3, x + 3, y + 3)
        root.after(16, move, step + 1)

    move()


def animate_network(avg_probs, winners=None, step=0):
    """Pulse through layers one at a time, send signal particles, then
    highlight ALL predicted digits in the output layer.
    avg_probs : averaged softmax across all digits (for colour intensity)
    winners   : list of predicted digit ints (one per drawn digit)
    """
    if winners is None:
        winners = [int(np.argmax(avg_probs))]

    # ── RESET all nodes and lines ──────────────────────────────────────
    for layer in network_nodes:
        for node in layer:
            nn_canvas.itemconfig(node, fill="#1a1a2e", outline=TEXT_DIM, width=1)
    for layer_lines in network_lines:
        for line in layer_lines:
            nn_canvas.itemconfig(line, fill="#1a1a2e")

    best = int(np.argmax(avg_probs))

    # ── INPUT node ────────────────────────────────────────────────────
    input_node = network_nodes[0][0]
    nn_canvas.itemconfig(input_node, fill=ACCENT_CYAN, outline=ACCENT_CYAN)
    pulse_node(input_node, ACCENT_CYAN)

    # ── HIDDEN layers ────────────────────────────────────────────────
    for layer in network_nodes[1:-1]:
        for node in layer:
            import random as _rng
            strength = _rng.random()
            if strength > 0.45:
                color = ACCENT_CYAN
            else:
                color = ACCENT_PURPLE
            nn_canvas.itemconfig(node, fill=color, outline=color)
            pulse_node(node, color)

    # ── OUTPUT layer ─────────────────────────────────────────────────
    for i, node in enumerate(network_nodes[-1]):
        if i in winners:
            nn_canvas.itemconfig(node, fill=ACCENT_CYAN, outline="white", width=2)
            pulse_node(node, ACCENT_CYAN)
        else:
            intensity = max(30, int(avg_probs[i] * 255))
            color = f"#{intensity:02x}00aa"
            nn_canvas.itemconfig(node, fill=color, outline=color)

    # ── SIGNAL PARTICLES along every connection ───────────────────────
    for coords in signal_particles:
        x1, y1, x2, y2 = coords
        animate_signal(x1, y1, x2, y2, ACCENT_CYAN)


def reset_network():
    for layer in network_nodes:
        for node in layer:
            nn_canvas.itemconfig(node, fill="#1a1a2e", outline=TEXT_DIM, width=1)
    for layer_lines in network_lines:
        for line in layer_lines:
            nn_canvas.itemconfig(line, fill="#1a1a2e")



# ==============================
# SCAN EFFECT HELPERS
# ==============================

scan_line_id = [None]
scan_active  = [False]

def run_scan_effect(y=0):
    if not scan_active[0]:
        return
    if scan_line_id[0]:
        canvas.delete(scan_line_id[0])
    scan_line_id[0] = canvas.create_line(
        0, y, canvas_size, y,
        fill=ACCENT_CYAN, width=2
    )
    if y < canvas_size:
        root.after(8, run_scan_effect, y + 14)
    else:
        canvas.delete(scan_line_id[0])
        scan_active[0] = False

def start_scan():
    scan_active[0] = True
    run_scan_effect(0)


# ==============================
# SHARED PREPROCESSING  (multi-digit)
# ==============================

def run_preprocessing():
    """Returns list of (shaped, norm) tuples sorted left→right.
    Falls back to None, None if canvas is blank."""
    raw = np.array(image)
    _, thresh = cv2.threshold(raw, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    # Sort contours left → right by X position
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    results = []
    for contour in contours:
        # Skip tiny noise
        if cv2.contourArea(contour) < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        digit_crop = thresh[y:y+h, x:x+w]
        size       = max(w, h) + 40
        square     = np.zeros((size, size), dtype=np.uint8)
        x_off      = (size - w) // 2
        y_off      = (size - h) // 2
        square[y_off:y_off+h, x_off:x_off+w] = digit_crop
        img28      = cv2.resize(square, (28, 28))
        norm       = img28 / 255.0
        shaped     = norm.reshape(1, 28, 28, 1)
        results.append((shaped, norm))

    if len(results) == 0:
        return None, None

    # For single-digit callers: return last (rightmost) shaped + norm
    # For multi-digit callers: use full results list via run_preprocessing_multi()
    return results[-1][0], results[-1][1]


def run_preprocessing_multi():
    """Returns sorted list of (shaped, norm) for every detected digit."""
    raw = np.array(image)
    _, thresh = cv2.threshold(raw, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        digit_crop = thresh[y:y+h, x:x+w]
        size       = max(w, h) + 40
        square     = np.zeros((size, size), dtype=np.uint8)
        x_off      = (size - w) // 2
        y_off      = (size - h) // 2
        square[y_off:y_off+h, x_off:x_off+w] = digit_crop
        img28      = cv2.resize(square, (28, 28))
        norm       = img28 / 255.0
        shaped     = norm.reshape(1, 28, 28, 1)
        results.append((shaped, norm))
    return results


# ==============================
# REAL-TIME PREDICTION
# ==============================

def realtime_predict():
    if is_drawing:
        return
    digits_data = run_preprocessing_multi()
    if not digits_data:
        return

    recognized = []
    for shaped, img28 in digits_data:
        pred  = model.predict(shaped, verbose=0)[0]
        recognized.append((pred, shaped, img28))

    # Build final number string (left → right order)
    final_number = "".join(str(int(np.argmax(p))) for p, _, __ in recognized)

    # Average all predictions across all digits for the graph/bars
    avg_pred   = np.mean([p for p, _, __ in recognized], axis=0)

    # Most recently drawn digit (rightmost) for preview
    last_pred, last_shaped, last_img28 = recognized[-1]
    last_digit      = int(np.argmax(last_pred))
    last_confidence = float(np.max(last_pred)) * 100

    # Overall confidence = average of each digit's top confidence
    avg_confidence = float(np.mean([np.max(p) for p, _, __ in recognized])) * 100

    digit_var.set(final_number)
    conf_var.set(f"CONF:  {avg_confidence:.2f}%")
    status_var.set(f"live  ·  {len(recognized)} digit(s) found ✓")

    # Bars + graph show AVERAGED probabilities across all digits
    update_prob_bars(avg_pred)
    bar_bg.coords(conf_bar, 0, 0, int(500 * avg_confidence / 100), 10)
    update_graph(avg_pred)

    # Preview + Grad-CAM show the LAST (most recently drawn) digit
    update_preview(last_img28)
    _winners = [int(np.argmax(p)) for p, _, __ in recognized]
    animate_network(avg_pred, _winners)

    # Top-3 from averaged probs
    top3  = np.argsort(avg_pred)[::-1][:3]
    lines = [f"TOP  [ {len(recognized)} digit(s) ]"]
    for rank, idx in enumerate(top3):
        lines.append(f"  #{rank+1}  digit {idx}  {avg_pred[idx]*100:.1f}%")
    topn_var.set("\n".join(lines))

    # Update history
    prediction_history.append(final_number)
    if len(prediction_history) > 5:
        prediction_history.pop(0)
    history_label.config(text="  →  ".join(prediction_history))


# ==============================
# PREDICT + CLEAR (now also update graph)
# ==============================

def predict_digit():
    status_var.set("analyzing ...")
    digit_var.set("?")
    conf_var.set("CONF:  --.--%")
    root.update()

    # AI Scan effect on canvas
    start_scan()

    # Multi-digit preprocessing
    digits_data = run_preprocessing_multi()
    if not digits_data:
        status_var.set("awaiting input ...")
        return

    recognized = []
    for shaped, img28 in digits_data:
        # DEBUGGING TRICK 1 — print image shape
        print("Image Shape:", shaped.shape)

        # DEBUGGING TRICK 2 — save processed image
        cv2.imwrite("processed.png", shaped[0] * 255)

        prediction = model.predict(shaped, verbose=0)

        # DEBUGGING TRICK 3 — print prediction probabilities
        print("Prediction Probabilities:")
        print(prediction)

        recognized.append((prediction[0], shaped, img28))

    # Build final number string (left → right)
    final_number = "".join(str(int(np.argmax(p))) for p, _, __ in recognized)

    # Average all predictions across all digits for the graph/bars
    avg_pred       = np.mean([p for p, _, __ in recognized], axis=0)

    # Most recently drawn digit (rightmost) for preview
    last_pred, last_shaped, last_img28 = recognized[-1]
    last_digit      = int(np.argmax(last_pred))

    # Overall confidence = average of each digit's top confidence
    avg_confidence = float(np.mean([np.max(p) for p, _, __ in recognized])) * 100

    digit_var.set(final_number)
    conf_var.set(f"CONF:  {avg_confidence:.2f}%")
    status_var.set(f"{len(recognized)} digit(s) identified ✓")

    flash_digit(last_digit)
    animate_conf_bar(int(500 * avg_confidence / 100))

    # Bars + graph show AVERAGED probabilities across all digits
    update_prob_bars(avg_pred)
    update_graph(avg_pred)

    # Preview + Grad-CAM show the LAST (most recently drawn) digit
    update_preview(last_img28)
    _winners = [int(np.argmax(p)) for p, _, __ in recognized]
    animate_network(avg_pred, _winners)

    # Top-3 from averaged probs
    top3 = np.argsort(avg_pred)[::-1][:3]
    lines = [f"TOP  [ {len(recognized)} digit(s) ]"]
    for rank, idx in enumerate(top3):
        lines.append(f"  #{rank+1}  digit {idx}  {avg_pred[idx]*100:.1f}%")
    topn_var.set("\n".join(lines))

    # Store prediction history (last 5)
    prediction_history.append(final_number)
    if len(prediction_history) > 5:
        prediction_history.pop(0)
    history_label.config(text="  →  ".join(prediction_history))


def flash_digit(digit, flashes=6, count=0):
    if count < flashes:
        color = ACCENT_CYAN if count % 2 == 0 else TEXT_DIM
        digit_display.config(fg=color)
        root.after(60, flash_digit, digit, flashes, count + 1)
    else:
        digit_display.config(fg=ACCENT_CYAN)

def animate_conf_bar(target_w, current=0):
    step = max(1, (target_w - current) // 4)
    current = min(current + step, target_w)
    bar_bg.coords(conf_bar, 0, 0, current, 10)
    if current < target_w:
        root.after(16, animate_conf_bar, target_w, current)

def clear_canvas():
    canvas.delete("all")
    draw_grid()
    canvas.create_text(canvas_size // 2, canvas_size // 2,
                       text="[ draw here ]", font=label_font, fill=TEXT_DIM)
    draw_pil.rectangle([0, 0, canvas_size, canvas_size], fill=255)

    digit_var.set("?")
    conf_var.set("CONF:  --.--%")
    status_var.set("awaiting input ...")
    bar_bg.coords(conf_bar, 0, 0, 0, 10)
    for i in range(10):
        x0 = start_x + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        bar_area.coords(prob_bars[i], x0, 4 + bar_max_h, x1, 4 + bar_max_h)
        bar_area.itemconfig(prob_bars[i], fill=ACCENT_PURPLE)
        bar_area.itemconfig(prob_labels[i], fill=TEXT_DIM)

    reset_graph()
    topn_var.set("")
    prediction_history.clear()
    history_label.config(text="")
    clear_preview()
    reset_network()
    scan_active[0] = False




# ==============================
# START GUI LOOP
# ==============================

root.mainloop()