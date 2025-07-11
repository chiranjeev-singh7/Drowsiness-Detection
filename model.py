import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create output folder
os.makedirs("models", exist_ok=True)

# Load datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(24, 24),
    batch_size=32,
    shuffle=True,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(24, 24),
    batch_size=32,
    shuffle=True,
)

print("Classes:", train_data.class_names)

normalization_layer = layers.Rescaling(1.0/255)
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(AUTOTUNE)
val_data = val_data.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

model.save("models/eye_model.keras")
