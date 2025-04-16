import numpy as np
import seaborn as sns

from keras import layers
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='grey')
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = keras.Sequential(
    [
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
)
model.summary()
# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=32)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"\nTocnost(trening skup): {train_acc:.2f}")
print(f"\nTocnost(test skup): {test_acc:.2f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_pred_probs = model.predict(x_test_s)
y_pred = np.argmax(y_pred_probs, axis=1)
matrica_zabune = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(matrica_zabune, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Test vrijednosti")
plt.ylabel("Stvarne vrijednosti")
plt.title("Matrica")
plt.show()


# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
incorrect_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(10, 7))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}\nTest: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
