import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model_final.h5')

# Class names
Classnames = ['Bear', 'Black Sea Sprat', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Gilt Head Bream', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Horse Mackerel', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red Mullet', 'Red Sea Bream', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea Bass', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Striped Red Mullet', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Trout', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra', 'antelope', 'badger', 'bat', 'bee', 'beetle', 'bison', 'boar', 'cane', 'cat', 'cavallo', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crow', 'dog', 'dolphin', 'donkey', 'dragonfly', 'elefante', 'farfalla', 'flamingo', 'fly', 'gallina', 'gatto', 'gorilla', 'grasshopper', 'hare', 'hornbill', 'hummingbird', 'hyena', 'ladybugs', 'lobster', 'mosquito', 'moth', 'mucca', 'octopus', 'okapi', 'orangutan', 'ox', 'oyster', 'pecora', 'pelecaniformes', 'pigeon', 'porcupine', 'possum', 'ragno', 'rat', 'reindeer', 'sandpiper', 'scoiattolo', 'seal', 'wolf', 'wombat']

# Set the path to your test image
test_image_path = "image.png"

# Load and preprocess the test image
img_height = 224  # MobileNetV2 default input size
img_width = 224

# Load the image and resize it
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_height, img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Normalize the image
normalization_layer = layers.Rescaling(1./255)
img_array = normalization_layer(img_array)

# Make predictions on the single test image
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]  # Get the predicted class

# Display the image and prediction
plt.imshow(img)
plt.title(f"Predicted Class: {Classnames[predicted_class]}")
plt.axis("off")
plt.show()
