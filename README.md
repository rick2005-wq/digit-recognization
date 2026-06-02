# Neural Vision Interface ✨

An interactive handwritten digit recognition web application featuring a stunning **cyberpunk-inspired UI**, real-time CNN intermediate activation visualization, and dynamic animated data flows built entirely on top of **Streamlit**.

---

## 🚀 Features

- **Futuristic Deep Web UI**: Pure cyberpunk aesthetics with custom CSS, Orbitron fonts, glassmorphism panels, and scanline overlays.
- **Real-time Recognition**: Draw digits and predict them instantly via a highly accurate internal CNN model.
- **Live Neural Architecture Visualizer**: An advanced JS/SVG-powered neural network map that illustrates *exactly* what the CNN is seeing.
  - Features real-time activation magnitudes on hidden layers.
  - Pulsing animated signal flow (particles) along the connections.
  - Intelligent output node highlighting (unselected nodes dim, winning nodes burst with light).
- **Infinite Prediction History**: Scrollable tape of your session's drawing history, always available underneath the drawing canvas.
- **Interactive Confidence Analytics**: Plotly-powered probabilistic probability charts.
- **Robust MNIST Preprocessing**: Custom backend pipeline scales strokes automatically into strict 20x20 bounding boxes using a center-of-mass algorithm to match training accuracy exactly.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, HTML/CSS/JS (embedded)
- **Deep Learning**: TensorFlow / Keras
- **Image Processing**: OpenCV, Pillow (PIL)
- **Data & Math**: NumPy
- **Charting**: Plotly

---

## 📂 Project Structure

```text
Neural-Vision-Interface/
│
├── model/
│   └── digit_model.keras       # Trained CNN model
│
├── app_streamlit.py            # Main Streamlit Cyberpunk Web Application
├── train.py                    # Script to generate the CNN model
├── requirements.txt            # Dependency list
├── README.md                   # Documentation
└── .gitignore
```

---

## 💻 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rick2005-wq/digit-recognization.git
   cd digit-recognization
   ```

2. **Install dependencies:**
   Make sure you are working in an active Python virtual environment!
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application:**
   ```bash
   streamlit run app_streamlit.py
   ```

4. **Train your own model (Optional):**
   If you want to view the training accuracy and loss charts, simply run:
   ```bash
   python train.py
   ```

---

## ☁️ Deploy to Streamlit Community Cloud

Hosting this application live on the web is incredibly simple and entirely free:
1. Ensure your latest commits are pushed to this GitHub repository.
2. Visit [Streamlit Community Cloud (share.streamlit.io)](https://share.streamlit.io) and log in with your GitHub account.
3. Click the **"New app"** button.
4. Fill in the data:
   * **Repository:** `rick2005-wq/digit-recognization`
   * **Branch:** `main`
   * **Main file path:** `app_streamlit.py`
5. Click **Deploy!** Streamlit handles everything automatically from `requirements.txt`.

---

## 🔭 Future Concepts
- Multi-digit sequential OCR
- Extended support for EMNIST datasets
- Grad-CAM direct overlay visualizations
- Adding native live webcam processing directly into the app structure

**Author:** Debarghya
**License:** MIT License
