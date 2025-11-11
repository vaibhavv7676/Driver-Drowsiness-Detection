import cv2
import dlib
import numpy as np
import gradio as gr
import os
import urllib.request
from scipy.spatial import distance as dist
import soundfile as sf
import numpy as np
import io

# -----------------------------
# Model Setup
# -----------------------------
MODEL_PATH = "Files/shape_predictor_68_face_landmarks.dat"
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

if not os.path.exists(MODEL_PATH):
    os.makedirs("Files", exist_ok=True)
    print("🔽 Downloading dlib facial landmark model (~100MB compressed)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, "Files/shape_predictor_68_face_landmarks.dat.bz2")
        import bz2
        with bz2.BZ2File("Files/shape_predictor_68_face_landmarks.dat.bz2") as fr, open(MODEL_PATH, "wb") as fw:
            fw.write(fr.read())
        print("✅ Model ready at:", MODEL_PATH)
    except Exception as e:
        print(f"Error downloading or extracting model: {e}")
        # The app might fail if the model isn't there, but we proceed for clarity
        pass 

# -----------------------------
# Parameters
# -----------------------------
detector = dlib.get_frontal_face_detector()
# Only initialize if file exists to prevent hard crash if download failed
if os.path.exists(MODEL_PATH):
    predictor = dlib.shape_predictor(MODEL_PATH)
else:
    # Use a dummy function if predictor isn't available
    predictor = lambda *args: None

EYE_AR_THRESH_DROWSY = 0.26
EYE_AR_THRESH_SLEEP = 0.21
EYE_AR_CONSEC_FRAMES_DROWSY = 12
EYE_AR_CONSEC_FRAMES_SLEEP = 25

frame_counter = 0
frame_skip = 2
_frame_index = 0

# -----------------------------
# Helper Functions (Tone Generator for Web Alert)
# -----------------------------
def generate_alert_sound(duration=0.5, freq=440, volume=0.5, samplerate=22050):
    """Generates a simple sine wave tone for Gradio (numpy type)."""
    t = np.linspace(0, duration, int(samplerate * duration), False)
    data = volume * np.sin(2. * np.pi * freq * t)
    
    # Return (sample_rate, numpy_data) which Gradio expects for type="numpy"
    return samplerate, data.astype(np.float32)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame_bgr):
    global frame_counter
    
    # Handle case where predictor might not be initialized due to failed download
    if predictor is None:
        return frame_bgr, "MODEL MISSING ❌", None 

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector(gray)
    status = "Awake 😃"
    status_color = (0, 200, 0)
    alert_active = False
    sound_output = None 
    
    if not faces:
        status = "Face not found"

    for face in faces:
        landmarks = predictor(gray, face)
        pts = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)
        left_eye  = pts[36:42]
        right_eye = pts[42:48]
        left_ear  = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        cv2.polylines(frame_bgr, [left_eye],  True, (0, 255, 255), 1)
        cv2.polylines(frame_bgr, [right_eye], True, (0, 255, 255), 1)

        if ear < EYE_AR_THRESH_DROWSY:
            frame_counter += 1
        else:
            frame_counter = 0

        if frame_counter >= EYE_AR_CONSEC_FRAMES_SLEEP:
            status = "Sleeping 😴"
            status_color = (0, 0, 150)
            alert_active = True
        elif frame_counter >= EYE_AR_CONSEC_FRAMES_DROWSY:
            status = "Drowsy 💤"
            status_color = (0, 0, 255)
            alert_active = True

        # Draw status and EAR value
        cv2.putText(frame_bgr, f"Status: {status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        cv2.putText(frame_bgr, f"EAR: {ear:.2f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame_bgr, "LIVE", (frame_bgr.shape[1]-120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    if alert_active:
        # Draw flashing alert box
        cv2.rectangle(frame_bgr, (0,0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0,0,255), 50)
        cv2.putText(frame_bgr, "⚠ ALERT: DRIVER DROWSY", (40,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 4)
        
        # Generate sound (Gradio handles playback)
        sound_output = generate_alert_sound()
        
    return frame_bgr, status, sound_output

def process_stream(frame_rgb):
    global _frame_index
    
    # Return default status if no frame is received
    if frame_rgb is None:
        return None, "Awaiting Input...", None
    
    _frame_index += 1
    # Skip frames to reduce computational load
    if _frame_index % frame_skip != 0:
        return frame_rgb, "Active 😃", None 
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out_bgr, status, sound = detect_drowsiness(frame_bgr)
    
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), status, sound

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="🚗 Driver Drowsiness Detection") as demo:
    gr.Markdown(
        """
        # 🚗 Driver Drowsiness Detection  
        Real-time **live monitoring** optimized for desktop and mobile webcams.
        """
    )
    
    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="Web Live Frame", # Your requested frame label
            image_mode="RGB",
            height=320,
            width=480
        )
        with gr.Column(min_width=200):
            gr.Markdown("## Live Monitoring")
            status_output = gr.Textbox(label="Driver State", value="Awaiting Input...")
            
            # Audio component set to type="numpy" and invisible
            audio_output = gr.Audio(label="Alert Sound", type="numpy", interactive=False, visible=False) 

    # Link the stream function to the components
    webcam.stream(
        fn=process_stream, 
        inputs=webcam, 
        outputs=[webcam, status_output, audio_output]
    )

# -----------------------------
# Launch (CRITICAL FIX FOR RENDER PORT ISSUE using Gunicorn)
# -----------------------------
# We must expose the Gradio application object (demo.app) for Gunicorn to run.
# No need to call demo.launch() here.
app = demo.app
