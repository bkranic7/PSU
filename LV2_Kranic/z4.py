import numpy as np
import matplotlib.pyplot as plt

def checkerboard(square_size, rows, cols):
    board = np.ones((rows * square_size, cols * square_size))*255  # Početno crno polje
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:  # Postavljanje BIJELIH kvadrata
                board[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size] = 0
    return board

img = checkerboard(50, 4, 5)  # 4x4 polje, kvadrati 50px

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('on')  # Uklanjanje osi
plt.show()
