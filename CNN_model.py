import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
data_path = "./Dataset_BUSI_with_GT"

# Parameters
image_size = (128, 128)  # Resize images to 128x128
classes = ["benign", "malignant", "normal"]
# Data preprocessing
X, y = [], []
for idx, category in enumerate(classes):
    category_path = os.path.join(data_path, category)
    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, image_size)
            X.append(img_resized)
            y.append(idx)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X).reshape(-1, image_size[0], image_size[1], 1) / 255.0  # Normalize pixel values to [0, 1]
y = np.array(y)
y = to_categorical(y, num_classes=len(classes))

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# CNN Model creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save("BREAST_cancer_PREDICTION_model.h5")

print("Model training complete. Model saved as 'BREAST_cancer_PREDICTION_model.h5'.")