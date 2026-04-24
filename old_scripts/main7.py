import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
import os
import cv2

# Load model
model = tf.keras.models.load_model('animal_classification_model_final.h5')

# Load animal info
with open('animal_info.json', 'r') as f:
    animal_data = json.load(f)

# Animal class names
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

img_height = 224
img_width = 224
file_path = None

# Predict from image array (OpenCV or uploaded)
def predict_from_image(img_np, is_bgr=True):
    if is_bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_np)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    img_resized = cv2.resize(img_np, (img_height, img_width))
    img_array = tf.convert_to_tensor(img_resized, dtype=tf.float32)
    img_array = tf.expand_dims(img_array, 0)
    img_array = layers.Rescaling(1. / 255)(img_array)

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = tf.reduce_max(predictions, axis=1).numpy()[0]

    predicted_animal = Classnames[predicted_class]
    result_label.configure(
        text=f"🧠 Prediction: {predicted_animal}\n🎯 Confidence: {confidence:.2f}", fg="darkgreen"
    )

    for animal in animal_data['animals']:
        if animal['name'].lower() == predicted_animal.lower():
            details = animal['details']
            info_text = "\n".join([f"🔸 {k}: {v}" for k, v in details.items()])
            break
    else:
        info_text = "No extra info found."

    info_label.configure(text=info_text)

# Upload image from file
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        predict_from_image(img)

# 📷 Open webcam
def capture_from_camera():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera - Press SPACE to Capture, ESC to Exit", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera - Press SPACE to Capture, ESC to Exit", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:  # Spacebar to capture
            cap.release()
            cv2.destroyAllWindows()
            predict_from_image(frame)
            break
        elif key % 256 == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            break

# 🌿 GUI Setup
root = tk.Tk()
root.title("Jungle Wildlife Detector")
root.geometry("750x740")
root.configure(bg="#d0f0c0")  # jungle green

# Header
header = tk.Frame(root, bg="#a4d7a7", height=60)
header.pack(fill="x")
title = tk.Label(header, text="🦜 Jungle Wildlife Detector", font=("Arial", 20, "bold"), bg="#a4d7a7", fg="#1a531b")
title.pack(pady=10)

tk.Frame(root, height=2, bg="#ffffff").pack(fill="x")

# Image preview
img_label = Label(root, text="📷 Upload or Capture an Animal", font=("Arial", 14), bg="#d0f0c0")
img_label.pack(pady=20)

# Buttons
btn_frame = tk.Frame(root, bg="#d0f0c0")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="📁 Upload Image", command=upload_image,
                       font=("Arial", 14), bg="#7dbd7d", fg="white", width=18)
upload_btn.grid(row=0, column=0, padx=10)

camera_btn = tk.Button(btn_frame, text="📸 Capture from Camera", command=capture_from_camera,
                       font=("Arial", 14), bg="#6dbf6d", fg="white", width=18)
camera_btn.grid(row=0, column=1, padx=10)

detect_btn = tk.Button(root, text="🔍 Identify Animal (Upload only)", command=lambda: identify_animal(),
                       font=("Arial", 14), bg="#4b924b", fg="white", width=30)
detect_btn.pack(pady=10)

# Result and Info
result_label = Label(root, text="", font=("Arial", 14), bg="#d0f0c0")
result_label.pack(pady=15)

info_label = Label(root, text="", font=("Arial", 12), wraplength=660, justify="left", bg="#d0f0c0")
info_label.pack(pady=10)

root.mainloop()
