import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Učitavanje slike
image = Image.open("tiger.png")
image_array = np.array(image)

# a) Povećanje osvetljenja slike
brightened_image = np.clip(image_array * 1.5, 0, 255).astype(np.uint8)

# b) Rotacija slike za 90 stepeni u smeru kazaljke na satu
rotated_image = np.rot90(image_array, k=3)

# c) Zrcaljenje slike (horizontalno)
mirrored_image = np.flip(image_array, axis=1)

# d) Smanjenje rezolucije slike 10 puta
small_image = image.resize((image.width // 10, image.height // 10), Image.LANCZOS)

# e) Prikaz samo druge četvrtine slike po širini (ostalo crno)
masked_image = np.zeros_like(image_array)
w_quarter = image_array.shape[1] // 4
masked_image[:, w_quarter:2*w_quarter, :] = image_array[:, w_quarter:2*w_quarter, :]

# Prikaz svih transformacija
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(image_array)
axes[0, 0].set_title("Originalna slika")

axes[0, 1].imshow(brightened_image)
axes[0, 1].set_title("Povećana svjetlina")

axes[0, 2].imshow(rotated_image)
axes[0, 2].set_title("Rotirano 90°")

axes[1, 0].imshow(mirrored_image)
axes[1, 0].set_title("Zrcaljena slika")

axes[1, 1].imshow(small_image)
axes[1, 1].set_title("Smanjena rezolucija")

axes[1, 2].imshow(masked_image)
axes[1, 2].set_title("Druga četvrtina slike")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
