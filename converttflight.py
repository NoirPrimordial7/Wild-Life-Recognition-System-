import tensorflow as tf

# Load the model from .h5 file
model = tf.keras.models.load_model('animal_classification_model_final.h5')

# Save it in SavedModel format
tf.saved_model.save(model,'D:/C,C++/PBL/Model/saved_model_format')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('D:/C,C++/PBL/Model/saved_model_format')
tflite_model = converter.convert()

# Save the TFLite model
with open('animal_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)
