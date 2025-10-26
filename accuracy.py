# accuracy.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
model = load_model("models/eye_model.keras")
print(model.input_shape)

# -------------------------------
# Load the trained model
# -------------------------------
model_path = os.path.join("models", "eye_model.keras")
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")
print("Model output shape:", model.output_shape)

# -------------------------------
# Load test data
# -------------------------------
test_dir = os.path.join("data", "test")

# The target size must match the input size used during training (update if needed)
IMG_SIZE = (24, 24)  # <-- change if your training used a different size

datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    color_mode='grayscale',   # use 'rgb' if your model expects 3 channels
    class_mode='binary',
    shuffle=False
)

# -------------------------------
# Make predictions
# -------------------------------
y_pred_probs = model.predict(test_generator)

# For softmax output (2 neurons)
if y_pred_probs.shape[1] == 2:
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    # For sigmoid output (1 neuron)
    y_pred = (y_pred_probs > 0.5).astype(int).ravel()

y_true = test_generator.classes

# -------------------------------
# Compute metrics
# -------------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.title('Confusion Matrix - Drowsiness Detection')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()



