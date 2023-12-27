import numpy as np
import matplotlib.pyplot as plt

# Set the size of the plot window, maximum iterations, and the x and y limits of the plot
width, height = 800, 800
max_iter = 100
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Create a grid of complex numbers (real and imaginary parts)
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize an array to store the escape time (number of iterations)
escape_time = np.zeros(Z.shape, dtype=int)

# Calculate the Mandelbrot fractal
C = np.copy(Z)
for i in range(max_iter):
    mask = np.abs(Z) < 2  # Points that are still in the set (magnitude < 2)
    escape_time[mask] = i  # Update escape time for points still in the set
    Z[mask] = Z[mask]**2 + C[mask]  # Apply the Mandelbrot iteration

# Plot the fractal
plt.figure(figsize=(8, 8))
plt.imshow(escape_time, extent=[x_min, x_max, y_min, y_max])
plt.colorbar(label='Number of iterations')
plt.title('Mandelbrot Set')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('fractal.pdf')
