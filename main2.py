import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
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

# App functionality
def upload_image():
    global file_path, img_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
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
    result_label.configure(text=f"Predicted: {Classnames[predicted_class]} (Confidence: {confidence:.2f})")

# Initialize the app
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("600x500")

file_path = None

# Layout
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)

identify_button = tk.Button(root, text="Identify Animal", command=identify_animal, font=("Arial", 14))
identify_button.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

# Run the app
root.mainloop()
