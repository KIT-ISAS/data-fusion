import numpy as np
import matplotlib.colors
from matplotlib.patches import Ellipse


def plot_covariance_ellipse(axes, cov, color, linestyle="solid", fill_alpha=0.0):
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:2, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals[:2])
    axes.tick_params(axis='both', which='major')#, labelsize=20)

    fill_color_spec = list(matplotlib.colors.to_rgba(color))
    fill_color_spec[-1] = fill_alpha
    ellipse = Ellipse([0,0], w, h, theta, linestyle=linestyle, linewidth=1.5, facecolor=fill_color_spec, edgecolor=color)
    axes.add_patch(ellipse)