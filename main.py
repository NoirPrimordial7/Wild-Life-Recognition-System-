import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Set the path to your dataset
train_dir = r"C:\Users\ADITYA\.cache\kagglehub\datasets\antoreepjana\animals-detection-images-dataset\versions\7\train"

# Load the dataset
batch_size = 32
img_height = 224  # MobileNetV2 default input size
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Preprocess the data
normalization_layer = layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Load the MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create the new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 33  # Adjust the number of epochs as needed
history = model.fit(normalized_train_ds, epochs=epochs)

# Save the model
model.save('animal_classification_model_final.h5')

# Print class names for reference
print("Class names:", train_ds.class_names)

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()