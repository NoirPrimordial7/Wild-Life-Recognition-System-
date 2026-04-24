import tkinter as tk
from tkinter import filedialog, Label, Scrollbar, Frame, Text
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json

# Load model
model = tf.keras.models.load_model('animal_classification_model_final.h5')

# Load JSON animal info
with open('animal_info.json', 'r') as f:
    animal_data = json.load(f)

# Class labels
Classnames = [...]  # same long list as before (truncate here to save space)

img_height, img_width = 224, 224
file_path = None

# App setup
root = tk.Tk()
root.title("🐾 Wildlife Classifier")
root.geometry("800x650")
root.configure(bg="#f5f5f5")

# Fonts and styles
TITLE_FONT = ("Segoe UI", 20, "bold")
BUTTON_FONT = ("Segoe UI", 12)
LABEL_FONT = ("Segoe UI", 12)
RESULT_FONT = ("Segoe UI", 14, "bold")

# Upload image
def upload_image():
    global file_path, img_label
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        result_label.config(text="")  # Clear previous
        info_text.config(state="normal")
        info_text.delete("1.0", tk.END)
        info_text.config(state="disabled")

# Predict animal
def identify_animal():
    if not file_path:
        result_label.config(text="No image uploaded!", fg="red")
        return

    # Preprocess
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = layers.Rescaling(1. / 255)(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = tf.reduce_max(predictions, axis=1).numpy()[0]

    predicted_animal = Classnames[predicted_class]
    result_label.config(text=f"🧠 Predicted: {predicted_animal} ({confidence:.2f})", fg="green")

    # Display info
    info_text.config(state="normal")
    info_text.delete("1.0", tk.END)
    for animal in animal_data['animals']:
        if animal['name'].lower() == predicted_animal.lower():
            details = animal['details']
            formatted = "\n".join([f"{key}: {value}" for key, value in details.items()])
            info_text.insert(tk.END, formatted)
            break
    else:
        info_text.insert(tk.END, "Information not available.")

    info_text.config(state="disabled")

# ========== UI Layout ==========

# Title
title = tk.Label(root, text="🐾 Wildlife Detection System", font=TITLE_FONT, bg="#f5f5f5")
title.pack(pady=10)

# Image display
img_label = tk.Label(root, bg="#e0e0e0", width=300, height=300)
img_label.pack(pady=10)

# Buttons
btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="📤 Upload Image", command=upload_image, font=BUTTON_FONT, bg="#4caf50", fg="white", width=18)
upload_btn.grid(row=0, column=0, padx=10)

identify_btn = tk.Button(btn_frame, text="🔍 Identify Animal", command=identify_animal, font=BUTTON_FONT, bg="#2196f3", fg="white", width=18)
identify_btn.grid(row=0, column=1, padx=10)

# Result label
result_label = tk.Label(root, text="", font=RESULT_FONT, bg="#f5f5f5")
result_label.pack(pady=10)

# Info panel with scroll
info_frame = Frame(root, bg="#ffffff", bd=2, relief="groove")
info_frame.pack(fill="both", expand=True, padx=30, pady=10)

scrollbar = Scrollbar(info_frame)
scrollbar.pack(side="right", fill="y")

info_text = Text(info_frame, font=LABEL_FONT, wrap="word", yscrollcommand=scrollbar.set, bg="#ffffff", relief="flat", state="disabled")
info_text.pack(fill="both", expand=True, padx=10, pady=10)
scrollbar.config(command=info_text.yview)

# Run
root.mainloop()
