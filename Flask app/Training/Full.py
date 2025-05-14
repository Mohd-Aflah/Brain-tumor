
# Importing Required Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from warnings import filterwarnings
filterwarnings("ignore")

# Define colors for visualization
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)
# Define Labels
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Model 1: EfficientNet-Based Tumor Classification Model

# Loading and Preparing Data
image_size = 150
X_train, y_train = [], []

# Load Training Data
for label in labels:
    folder_path = os.path.join('DataSet', 'Training', label)
    for image_file in tqdm(os.listdir(folder_path)):
        img = cv2.imread(os.path.join(folder_path, image_file))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=101)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=101)

# Encode Labels
y_train = tf.keras.utils.to_categorical([labels.index(label) for label in y_train])
y_test = tf.keras.utils.to_categorical([labels.index(label) for label in y_test])

# Build EfficientNet Model
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
model = tf.keras.Sequential([
    effnet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

# Train Model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=12,
    batch_size=32,
    verbose=1,
    callbacks=[tensorboard, checkpoint, reduce_lr]
)

# Plot Training Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate Model
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification Report and Confusion Matrix
print("Classification Report:\n", classification_report(y_test_classes, pred_classes, target_names=labels))
cm = confusion_matrix(y_test_classes, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize Sample Predictions
def visualize_predictions(X, y_true, y_pred, labels, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        img = X[idx]
        true_label = labels[np.argmax(y_true[idx])]
        pred_label = labels[np.argmax(y_pred[idx])]
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test, y_test, pred, labels)

# Model 2: YOLO-Based Tumor Segmentation Model

from roboflow import Roboflow
rf = Roboflow(api_key="a7b1JkwHtDpLlkj509uo")
project = rf.workspace("iotseecs").project("brain-tumor-yzzav")
version = project.version(1)
dataset = version.download("yolov11")

# Define Class Names and Colors
class_names = ['tumor']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))
def plot_segmentation(image, polygons, labels):
    h, w, _ = image.shape
    for polygon_num, polygon in enumerate(polygons):
        class_name = class_names[int(labels[polygon_num])]
        color = colors[class_names.index(class_name)]

        # Denormalize the Polygon Points
        points = []
        for i in range(0, len(polygon), 2):
            x = int(float(polygon[i]) * w)
            y = int(float(polygon[i + 1]) * h)
            points.append([x, y])

        points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(image, [points], color=color)
        centroid_x = int(np.mean(points[:, 0, 0]))
        centroid_y = int(np.mean(points[:, 0, 1]))
        cv2.putText(image, class_name, (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load YOLO Pretrained Model
model.train(
    data="C:/Users/moham/OneDrive/Desktop/MAIN PROJECT/BRAIN TUMOR/Backend/FULL/BRAIN-TUMOR-1/data.yaml",
    epochs=20,
    imgsz=416,
    batch=16,
    half=True
)