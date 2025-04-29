import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# TODO: strukturiraj konvolucijsku neuronsku mrezu
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(10, activation='softmax')
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# TODO: definiraj callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# TODO: provedi treniranje mreze pomocu .fit()
checkpoint_callback = ModelCheckpoint(
'best_model.h5',
monitor='val_accuracy',
save_best_only=True,
mode='max',
verbose=1
)
history = model.fit(
x_train, y_train,
epochs=10,
batch_size=64,
validation_split=0.1,
callbacks=[tensorboard_callback, checkpoint_callback]
)

#TODO: Ucitaj najbolji model
model = tf.keras.models.load_model('best_model.h5')
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
print(f'Točnost na skupu podataka za učenje: {train_acc * 100:.2f}%')
print(f'Točnost na skupu podataka za testiranje: {test_acc * 100:.2f}%')
y_train_pred = np.argmax(model.predict(x_train), axis=1)
cm_train = confusion_matrix(y_train, y_train_pred)

y_test_pred = np.argmax(model.predict(x_test), axis=1)

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', xticklabels=range(10), yticklabels=range(10))
plt.title('Matrica zabune - Skup podataka za učenje')
plt.xlabel('Predviđene klase')
plt.ylabel('Realne klase')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', xticklabels=range(10), yticklabels=range(10))
plt.title('Matrica zabune - Skup podataka za testiranje')
plt.xlabel('Previđene klase')
plt.ylabel('Realne klase')
plt.show()