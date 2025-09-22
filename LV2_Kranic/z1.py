import numpy as np
import matplotlib.pyplot as plt

# Define the x and y coordinates of the points
x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

# Create the plot
plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='-', color='blue', markersize=8, linewidth=2)

# Add labels and title
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("Primjer")

# Set axis limits
plt.xlim(0, 4)
plt.ylim(0, 4)

# Show the plot
plt.grid(True)
plt.show()
