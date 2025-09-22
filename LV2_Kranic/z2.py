import numpy as np
import matplotlib.pyplot as plt

# Učitavanje podataka iz mtcars.csv (pretpostavka da se nalazi u istom folderu)
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

# Izdvajanje kolona
mpg = data[:, 0]  # Potrošnja goriva (miles per gallon)
hp = data[:, 1]   # Konjska snaga
wt = data[:, 2]   # Težina vozila
cyl = data[:, 3]  # Broj cilindara

# b) Scatter plot: Odnos između mpg i hp
plt.figure(figsize=(8,5))
plt.scatter(hp, mpg, color='blue', alpha=0.7)
plt.xlabel("Konjska snaga (hp)")
plt.ylabel("Potrošnja goriva (mpg)")
plt.title("Odnos između potrošnje goriva i konjske snage")
plt.grid(True)
plt.show()

# c) Scatter plot sa težinom vozila kao veličinom tačke
plt.figure(figsize=(8,5))
plt.scatter(hp, mpg, s=wt*50, color='green', alpha=0.6, edgecolors='black')
plt.xlabel("Konjska snaga (hp)")
plt.ylabel("Potrošnja goriva (mpg)")
plt.title("Odnos između potrošnje, snage i težine vozila")
plt.grid(True)
plt.show()

# d) Računanje minimalne, maksimalne i srednje vrednosti potrošnje (mpg)
mpg_min = np.min(mpg)
mpg_max = np.max(mpg)
mpg_mean = np.mean(mpg)

print(f"Minimalna potrošnja: {mpg_min:.2f} mpg")
print(f"Maksimalna potrošnja: {mpg_max:.2f} mpg")
print(f"Srednja potrošnja: {mpg_mean:.2f} mpg")

# e) Analiza samo za automobile sa 6 cilindara
six_cyl_cars = data[cyl == 6]
mpg_six = six_cyl_cars[:, 0]

mpg_six_min = np.min(mpg_six)
mpg_six_max = np.max(mpg_six)
mpg_six_mean = np.mean(mpg_six)

print(f"(6 cilindara) Minimalna potrošnja: {mpg_six_min:.2f} mpg")
print(f"(6 cilindara) Maksimalna potrošnja: {mpg_six_max:.2f} mpg")
print(f"(6 cilindara) Srednja potrošnja: {mpg_six_mean:.2f} mpg")

