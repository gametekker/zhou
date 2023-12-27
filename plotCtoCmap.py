import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#===INTUITION===
#f: z -> w
#z is the set of complex numbers, C
#arrange z into a plane
#apply f to the plane to make w
#for each point in w, move along "z" axis by the real component of the corresponding point in z

# Create a grid of complex numbers (z = x + iy)
x = np.linspace(-2, 2, 30)  # Real part of z
y = np.linspace(-2, 2, 30)  # Imaginary part of z
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y
print(Z)

# Compute w = z^7
W = Z**7

# Print the shapes of the arrays
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Shape of Z:", Z.shape)
print("Shape of W:", W.shape)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot with real part of z on the 'z' axis, and real and imaginary parts of w on 'x' and 'y' axes
ax.plot_surface(np.real(W), np.imag(W), X, cmap='viridis')

ax.set_xlabel('Real Part of w (Re[w])')
ax.set_ylabel('Imaginary Part of w (Im[w])')
ax.set_zlabel('Real Part of z (Re[z])')
ax.set_title('3D Plot of w = z^2')

# Save the plot as an image (not as a 3D model)
plt.savefig('3d_plot.png')
plt.show()

#TODO: plot for different angles - for w=exp(z) it looks like a slinky bouncing as you animate across angles