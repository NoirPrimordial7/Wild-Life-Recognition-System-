import tkinter as tk
from tkinter import filedialog, Label, messagebox
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
    pil_import_error = None
except Exception as e:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False
    pil_import_error = e
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    numpy_import_error = None
except Exception as e:
    np = None
    NUMPY_AVAILABLE = False
    numpy_import_error = e
import json
import os

# Guarded imports for optional heavy dependencies
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    tf_import_error = None
except Exception as e:
    tf = None
    layers = None
    TF_AVAILABLE = False
    tf_import_error = e

try:
    import cv2
    CV2_AVAILABLE = True
    cv2_import_error = None
except Exception as e:
    cv2 = None
    CV2_AVAILABLE = False
    cv2_import_error = e

# Model will be loaded after MODEL_PATH is defined
model = None
model_load_error = None

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "animal_classification_model_final.h5")
INFO_PATH = os.path.join(BASE_DIR, "animal_info.json")

# ---------- Load model ----------
if TF_AVAILABLE:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        model = None
        model_load_error = e
else:
    model = None
    # model_load_error already set from tf_import_error when TF not available

# ---------- Load animal info ----------
try:
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        animal_data = json.load(f)
except Exception as e:
    animal_data = {"animals": []}
    print(f"Warning: could not load animal_info.json ({e}). Extra info will be blank.")

# ---------- Class names ----------
Classnames = ['Bear', 'Black Sea Sprat', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle',
              'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish',
              'Fox', 'Frog', 'Gilt Head Bream', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal',
              'Hedgehog', 'Hippopotamus', 'Horse', 'Horse Mackerel', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala',
              'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse',
              'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit',
              'Raccoon', 'Raven', 'Red Mullet', 'Red Sea Bream', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea Bass',
              'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow',
              'Spider', 'Squid', 'Squirrel', 'Starfish', 'Striped Red Mullet', 'Swan', 'Tick', 'Tiger', 'Tortoise',
              'Trout', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra', 'antelope', 'badger', 'bat',
              'bee', 'beetle', 'bison', 'boar', 'cane', 'cat', 'cavallo', 'chimpanzee', 'cockroach', 'cow',
              'coyote', 'crow', 'dog', 'dolphin', 'donkey', 'dragonfly', 'elefante', 'farfalla', 'flamingo', 'fly',
              'gallina', 'gatto', 'gorilla', 'grasshopper', 'hare', 'hornbill', 'hummingbird', 'hyena', 'ladybugs',
              'lobster', 'mosquito', 'moth', 'mucca', 'octopus', 'okapi', 'orangutan', 'ox', 'oyster', 'pecora',
              'pelecaniformes', 'pigeon', 'porcupine', 'possum', 'ragno', 'rat', 'reindeer', 'sandpiper',
              'scoiattolo', 'seal', 'wolf', 'wombat']

IMG_H = 224
IMG_W = 224
file_path = None  # last chosen file

# ---------- GUI ----------
root = tk.Tk()
root.title("Jungle Wildlife Detector")
root.geometry("750x740")
root.configure(bg="#d0f0c0")

header = tk.Frame(root, bg="#a4d7a7", height=60)
header.pack(fill="x")
title = tk.Label(header, text="🦜 Jungle Wildlife Detector", font=("Arial", 20, "bold"), bg="#a4d7a7", fg="#1a531b")
title.pack(pady=10)
tk.Frame(root, height=2, bg="#ffffff").pack(fill="x")

img_label = Label(root, text="📷 Upload or Capture an Animal", font=("Arial", 14), bg="#d0f0c0")
img_label.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 14), bg="#d0f0c0", fg="darkgreen")
result_label.pack(pady=15)

info_label = Label(root, text="", font=("Arial", 12), wraplength=660, justify="left", bg="#d0f0c0")
info_label.pack(pady=10)

# ---------- Startup dependency check ----------
def _startup_check():
    problems = []
    if not NUMPY_AVAILABLE:
        problems.append(f"NumPy not installed: {numpy_import_error}")
    if not PIL_AVAILABLE:
        problems.append(f"Pillow not installed: {pil_import_error}")
    if not TF_AVAILABLE:
        problems.append(f"TensorFlow not available: {tf_import_error}")
    if not CV2_AVAILABLE:
        problems.append(f"OpenCV (cv2) not available: {cv2_import_error}")
    if model is None:
        if model_load_error:
            problems.append(f"Model failed to load: {model_load_error}")
        else:
            problems.append(f"Model file not found or not loaded: {MODEL_PATH}")

    if problems:
        msg = "\n".join(problems)
        full = "Some optional dependencies or the model are missing/failed to load:\n\n" + msg + "\n\nYou can install missing packages with:\n  pip install -r requirements.txt"
        # show warning but allow the app to continue; buttons will show specific errors when used
        try:
            messagebox.showwarning("Startup issues detected", full)
        except Exception:
            print(full)

_startup_check()
# ---------- Inference helper ----------
def _predict_and_render(img_np_rgb):
    """img_np_rgb must be RGB uint8"""
    # preview
    # preview (optional)
    if PIL_AVAILABLE and Image is not None and ImageTk is not None:
        try:
            pil_img = Image.fromarray(img_np_rgb)
            preview = pil_img.copy()
            preview.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(preview)
            img_label.configure(image=img_tk, text="")
            img_label.image = img_tk
        except Exception:
            img_label.configure(text="(Preview unavailable)")
    else:
        img_label.configure(text="(Preview unavailable - install Pillow)")

    # preprocess
    if not CV2_AVAILABLE:
        # try to use TensorFlow resize if available
        if TF_AVAILABLE:
            try:
                x = tf.convert_to_tensor(img_np_rgb, dtype=tf.float32)
                x = tf.image.resize(x, (IMG_H, IMG_W))
                x = tf.expand_dims(x, 0)
                x = layers.Rescaling(1.0 / 255.0)(x)
            except Exception as e:
                result_label.configure(text=f"Error: preprocessing failed (no cv2): {e}")
                return
        else:
            result_label.configure(text="Error: OpenCV not available and TensorFlow resize unavailable.")
            return
    else:
        try:
            resized = cv2.resize(img_np_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x = tf.convert_to_tensor(resized, dtype=tf.float32)
            x = tf.expand_dims(x, 0)
            x = layers.Rescaling(1.0 / 255.0)(x)
        except Exception as e:
            result_label.configure(text=f"Error during preprocessing: {e}")
            return

    # ensure model loaded
    if model is None:
        result_label.configure(text="Error: model not loaded. See console or startup warning.")
        return

    # predict
    preds = model.predict(x, verbose=0)
    predicted_class = int(tf.argmax(preds, axis=1).numpy()[0])
    confidence = float(tf.reduce_max(preds, axis=1).numpy()[0])

    predicted_animal = Classnames[predicted_class] if 0 <= predicted_class < len(Classnames) else f"Class {predicted_class}"
    result_label.configure(text=f"🧠 Prediction: {predicted_animal}\n🎯 Confidence: {confidence:.2f}")

    # extra info
    info_text = "No extra info found."
    for animal in animal_data.get("animals", []):
        if animal.get("name", "").lower() == predicted_animal.lower():
            details = animal.get("details", {})
            info_text = "\n".join([f"🔸 {k}: {v}" for k, v in details.items()])
            break
    info_label.configure(text=info_text)

# ---------- Actions ----------
def predict_from_image(img_bgr):
    if img_bgr is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _predict_and_render(img_rgb)

def upload_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = None
        # prefer cv2 if available
        if CV2_AVAILABLE:
            try:
                img = cv2.imread(file_path)
            except Exception:
                img = None
        # fallback to Pillow
        if img is None and PIL_AVAILABLE and Image is not None:
            try:
                pil = Image.open(file_path).convert("RGB")
                img = np.array(pil)[:, :, ::-1]  # PIL RGB -> BGR for predict_from_image
            except Exception:
                img = None

        if img is None:
            messagebox.showerror("Error", "Could not load image. Install Pillow or OpenCV to enable image loading.")
            return

        predict_from_image(img)

def identify_animal():
    """Runs prediction using the last uploaded file."""
    if not file_path:
        messagebox.showinfo("Info", "Please upload an image first.")
        return
    img = cv2.imread(file_path)
    predict_from_image(img)

def capture_from_camera():
    if not CV2_AVAILABLE:
        messagebox.showerror("Camera Error", "OpenCV not available - camera capture disabled.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera (index 0).")
        return

    cv2.namedWindow("Camera - SPACE to Capture, ESC to Exit", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera - SPACE to Capture, ESC to Exit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            cap.release()
            cv2.destroyAllWindows()
            predict_from_image(frame)
            break
        elif key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- Buttons ----------
btn_frame = tk.Frame(root, bg="#d0f0c0")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="📁 Upload Image", command=upload_image,
                       font=("Arial", 14), bg="#7dbd7d", fg="white", width=18)
upload_btn.grid(row=0, column=0, padx=10)

camera_btn = tk.Button(btn_frame, text="📸 Capture from Camera", command=capture_from_camera,
                       font=("Arial", 14), bg="#6dbf6d", fg="white", width=18)
camera_btn.grid(row=0, column=1, padx=10)

detect_btn = tk.Button(root, text="🔍 Identify Animal (Upload only)", command=identify_animal,
                       font=("Arial", 14), bg="#4b924b", fg="white", width=30)
detect_btn.pack(pady=10)

root.mainloop()
