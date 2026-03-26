import tkinter as tk
from tkinter import filedialog, Label, Toplevel, Text
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
from tensorflow.keras import layers
import json
import os

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model_final.h5')

# Class names
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

# Image processing parameters
img_height = 224
img_width = 224

# Load animal info from JSON
with open("animal_info.json", "r") as f:
    animal_info = json.load(f)

# App functionality
def upload_image():
    global file_path, img_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = ImageOps.fit(img, (300, 300), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

def identify_animal():
    if not file_path:
        result_label.configure(text="No image uploaded!")
        return
    
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    normalization_layer = layers.Rescaling(1./255)
    img_array = normalization_layer(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = tf.reduce_max(predictions, axis=1).numpy()[0]

    # Display result
    result_label.configure(
        text=f"Predicted: {Classnames[predicted_class]} \nConfidence: {confidence:.2f}",
        fg="#5a189a"
    )

def show_info():
    if not result_label["text"]:
        result_label.configure(text="Identify an animal first!")
        return

    predicted_class = result_label["text"].split(":")[1].split("(")[0].strip()
    info = animal_info.get(predicted_class, "No information available for this animal.")
    
    # Open a new window
    info_window = Toplevel(root)
    info_window.title(f"About {predicted_class}")
    info_window.geometry("500x400")
    info_window.configure(bg="#D8E4BC")

    info_label = Label(info_window, text=f"Information about {predicted_class}", font=("Helvetica", 18, "bold"), bg="#D8E4BC", fg="#568203")
    info_label.pack(pady=10)

    info_text = Text(info_window, wrap="word", font=("Helvetica", 14), bg="#fff", fg="#333")
    info_text.insert("1.0", info)
    info_text.config(state="disabled")
    info_text.pack(pady=10, padx=20)

# Initialize the app
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("700x600")
root.configure(bg="#D8E4BC")

file_path = None

# Stylish Header
header = tk.Label(root, text="Welcome to Animal Classifier!", font=("Helvetica", 24, "bold"), bg="#D8E4BC", fg="#568203")
header.pack(pady=20)

# Upload Button
upload_button = tk.Button(
    root, 
    text="📤 Upload Image", 
    command=upload_image, 
    font=("Helvetica", 16), 
    bg="#568203", 
    fg="white",
    activebackground="#4e9a06", 
    bd=0, 
    relief="flat"
)
upload_button.pack(pady=20)

# Image Display
img_label = Label(root, bg="#D8E4BC")
img_label.pack(pady=10)

# Identify Button
identify_button = tk.Button(
    root, 
    text="🔍 Identify Animal", 
    command=identify_animal, 
    font=("Helvetica", 16), 
    bg="#568203", 
    fg="white",
    activebackground="#4e9a06", 
    bd=0, 
    relief="flat"
)
identify_button.pack(pady=20)

# Info Button
info_button = tk.Button(
    root,
    text="ℹ️ Animal Info",
    command=show_info,
    font=("Helvetica", 16),
    bg="#568203",
    fg="white",
    activebackground="#4e9a06",
    bd=0,
    relief="flat"
)
info_button.pack(pady=10)

# Result Display
result_label = Label(root, text="", font=("Helvetica", 18), bg="#D8E4BC", fg="#6a0572")
result_label.pack(pady=20)

# Footer
footer = tk.Label(root, text="✨ Powered by AI", font=("Helvetica", 12), bg="#D8E4BC", fg="#6c757d")
footer.pack(side="bottom", pady=10)

# Run the app
root.mainloop()
