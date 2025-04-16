import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Učitaj podatke
df = pd.read_csv('C:/Users/student/Desktop/LV5_Kranic/occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# a) Podjela podataka na train-test (80-20%), uz stratifikaciju
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# b) Skaliranje ulaznih podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c) Kreiranje i treniranje KNN modela (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# d) Predikcija i evaluacija modela
y_pred = knn.predict(X_test_scaled)

# Matrica zabune
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title("Matrica zabune")
plt.show()

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost klasifikacije: {accuracy:.4f}\n")
print("Izvještaj klasifikacije:\n", classification_report(y_test, y_pred, target_names=class_names))

# e) Utjecaj broja susjeda na točnost
k_values = range(1, 20)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(6,4))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel("Broj susjeda (K)")
plt.ylabel("Točnost")
plt.title("Utjecaj broja susjeda na točnost")
plt.show()

# f) Test bez skaliranja
knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = knn_no_scaling.predict(X_test)

accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f"Točnost bez skaliranja: {accuracy_no_scaling:.4f}")
