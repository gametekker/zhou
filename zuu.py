import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

def perturb_function(func, perturbation=0.1):
    """
    Perturbs the parameters of a given complex function slightly.
    This implementation is specific to the function structure.

    Args:
    func (callable): The original complex function (e.g., lambda z: 2**z**3 + 1j).
    perturbation (float): The magnitude of the perturbation.

    Returns:
    callable: A new function with slightly perturbed parameters.
    """

    # Example specific: Decompose and perturb the original function
    # For the function lambda z: 2**z**3 + 1j
    base = 2 + (random.random() - 0.5) * perturbation  # Perturb the base (2)
    constant = 1j + (random.random() - 0.5) * perturbation  # Perturb the constant (1j)

    # Return a new perturbed function
    return lambda z: base**z**3 + constant


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def blend_colormap_with_color(cmap_name, color, factor):
    """
    Blend a colormap with a specific color.

    cmap_name: The name of the colormap (e.g., 'inferno').
    color: A 3-tuple of RGB values (0-255) for the target color.
    factor: A value between 0 (original colormap) and 1 (full color).
    """
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))

    # Normalize color to [0, 1]
    color = np.array(color) / 255.0

    # Blend each color in the colormap with the target color
    blended_colors = (1 - factor) * colors[:, :3] + factor * color

    return LinearSegmentedColormap.from_list(f'{cmap_name}_blended', np.hstack((blended_colors, colors[:, 3:])))

def get_phase(x_min,x_max,y_min,y_max,resolution,func,val):
    # Create a grid of complex numbers (z = x + iy)
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Apply the complex function
    W = func(Z)

    # Normalize real and imaginary parts of W
    norm_real = (np.real(W) - np.real(W).min()) / (np.real(W).max() - np.real(W).min())
    norm_imag = (np.imag(W) - np.imag(W).min()) / (np.imag(W).max() - np.imag(W).min())

    # Calculate magnitude and phase (angle)
    phase = np.angle(W)
    return X,Y,phase

def get_dimmest_color(cmap_name,offset):
    """
    Find the dimmest color in a Matplotlib colormap.

    cmap_name: The name of the colormap (e.g., 'viridis').
    """
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))

    # Calculate luminance for each color
    luminance = 0.21 * colors[:, 0] + 0.72 * colors[:, 1] + 0.07 * colors[:, 2]

    # Find the index of the dimmest color
    dimmest_index = np.argmin(luminance)
    dimmest_color = colors[dimmest_index+offset, :3]

    dimmest_color = (dimmest_color * 255).astype(int)

    return tuple(dimmest_color)

def complex_function_autumn_hue_with_contours(func,
                                              x_min=-2,
                                              x_max=2,
                                              y_min=-2,
                                              y_max=2,
                                              resolution=500,
                                              colors='inferno',
                                              grad=5,
                                              offset=3):

    val = get_dimmest_color(colors,offset)
    print(val)

    # Plotting
    plt.figure(figsize=(6, 6))

    #Create background
    rgb_image=np.array([np.full((resolution, resolution), val[i]) for i in range(3)])
    print(rgb_image.shape)
    rgb_image=np.transpose(rgb_image,(1,2,0))

    # Plot hue-luminance with phase contours
    plt.subplot(1, 1, 1)
    plt.imshow(rgb_image, extent=[x_min, x_max, y_min, y_max])

    #dropshadow
    for i in range(grad):
        g=grad-i
        print(g)
        X,Y,phase=get_phase(x_min,x_max,y_min,y_max,resolution,func,val)
        plt.contour(X-.007*g,
                    Y-0.007*g,
                    phase,
                    20,
                    cmap=blend_colormap_with_color(colors,color=val,factor=.2+.02*g))  # Phase contour lines

    #main line
    X, Y, phase = get_phase(x_min, x_max, y_min, y_max, resolution, func, val)
    plt.contour(X, Y, phase, 20, cmap=colors)  # Phase contour lines

    plt.title(f'{colors[0].capitalize()}{colors[1:]} Hue-Luminance with Phase Contours')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    plt.savefig(f'{colors}.pdf')

# Example usage with a complex function
complex_function_autumn_hue_with_contours(lambda z: 1 + 2j*z + 1/(3j*np.exp(z)**2),colors='gist_heat',resolution=2000,grad=40,offset=0)
