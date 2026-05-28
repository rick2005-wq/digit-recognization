# ==============================
# IMPORT LIBRARIES
# ==============================

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ==============================
# LOAD MNIST DATASET
# ==============================

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


# ==============================
# CHECK DATA SHAPES
# ==============================

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ==============================
# NORMALIZE PIXEL VALUES
# ==============================

x_train = x_train / 255.0
x_test = x_test / 255.0


# ==============================
# RESHAPE DATA FOR CNN
# ==============================

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))


# ==============================
# BUILD CNN MODEL
# ==============================

model = models.Sequential()

model.add(layers.Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(
    64,
    (3, 3),
    activation='relu'
))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(
    64,
    activation='relu'
))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(
    10,
    activation='softmax'
))


# ==============================
# DISPLAY MODEL ARCHITECTURE
# ==============================

model.summary()


# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ==============================
# TRAIN MODEL
# ==============================

history = model.fit(
    x_train,
    y_train,
    epochs=60,
    batch_size=64,
    validation_data=(x_test, y_test)
)


# ==============================
# EVALUATE MODEL
# ==============================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)


# ==============================
# SAVE TRAINED MODEL
# ==============================

model.save("model/digit_model.keras")

print("\nModel saved successfully!")


# ==============================
# GENERATE PREDICTIONS
# ==============================

print("\nGENERATING GRAPHS...")

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='magma'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

print("Saved confusion_matrix.png")

plt.close()


# ==============================
# SAMPLE PREDICTIONS GRID
# ==============================

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')

    pred = np.argmax(y_pred[i])

    plt.title(f"Pred: {pred}")

    plt.axis('off')

plt.tight_layout()

plt.savefig("sample_predictions.png")

print("Saved sample_predictions.png")

plt.close()


# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='lower right')

plt.savefig("accuracy_graph.png")

print("Saved accuracy_graph.png")

plt.close()


# ==============================
# LOSS GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper right')

plt.savefig("loss_graph.png")

print("Saved loss_graph.png")

plt.close()


print("\nALL GRAPHS GENERATED SUCCESSFULLY")