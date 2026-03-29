import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision
import time

# --- Load PCA Logic from your existing step3 ---
def load_pca_model(path):
    data = np.load(path, allow_pickle=True)
    class_names = data['class_names'].tolist()
    model = {}
    for label in class_names:
        safe = label.replace(' ', '_')
        model[label] = {'mean': data[f"{safe}__mean"], 'eigenvecs': data[f"{safe}__eigenvecs"]}
    return model, class_names

def classify(vec, model, class_names):
    dists = {}
    for label in class_names:
        mu, U = model[label]['mean'], model[label]['eigenvecs']
        vc = vec - mu
        proj = U @ (U.T @ vc)
        dists[label] = float(np.linalg.norm(vc - proj))
    best = min(dists, key=dists.get)
    return best, dists[best]

def normalise(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    pts -= pts[0] # Center on wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    return (pts / scale).flatten()

class GestureApp:
    def __init__(self, window, model_path):
        self.window = window
        self.window.title("ASL Real-Time Recognition")
        
        # Load Model
        self.model, self.classes = load_pca_model(model_path)
        
        # UI Elements
        self.video_label = tk.Label(window)
        self.video_label.pack(padx=10, pady=10)
        
        self.result_var = tk.StringVar(value="Waiting for hand...")
        self.label_display = tk.Label(window, textvariable=self.result_var, font=("Helvetica", 24, "bold"), fg="#2ec99e")
        self.label_display.pack(pady=10)

        # Initialize Camera & MediaPipe
        self.cap = cv2.VideoCapture(0) # Change to 1 if 0 doesn't work
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE, num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)

            if result.hand_landmarks:
                label, dist = classify(normalise(result.hand_landmarks[0]), self.model, self.classes)
                if dist < 0.4: # Confidence threshold
                    self.result_var.set(f"MEANING: {label.upper()}")
                else:
                    self.result_var.set("Low Confidence...")
            else:
                self.result_var.set("Show your hand...")

            # Convert to Tkinter format
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_frame)

    def __del__(self):
        self.cap.release()

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root, "pca_model.npz")
    root.mainloop()