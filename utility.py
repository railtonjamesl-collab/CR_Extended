# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.spatial.distance import directed_hausdorff
def fwhm_to_sigma(fwhm):
    """
    Full width half maximum smoothing
    """
    if fwhm <= 0:
        return 0
    
    sigma = fwhm/(2 * np.sqrt(2 * np.log(2)))
    
    return sigma

def ball_signal(field_shape, radius, magnitude):
    """
    Genearte ball signal centered in the middle of the grid
    """
    # compute dimension
    dim = len(field_shape) 
    
    #extract coordinate grids
    coord = np.indices(field_shape) 
    
    # find center point (transalte by one unit by python convention)
    center = (np.array(field_shape) - 1)/2 
    
    #initialise the grid
    distance_square = np.zeros(field_shape)
    signal = np.zeros(field_shape)
    for axis in range(dim):
        
        distance_square += (center[axis] - coord[axis])**2
        
    signal[distance_square <= radius**2] = magnitude

    
    return(signal)

def gradient_signal(field_shape, min_sig, max_sig):
    
    """
    Generate a gradient signal, with increasing gradient over column axis i.e. last axis (y direction?)
    """
    # compute dimension
    dim = len(field_shape)
    
    #generating 1D gradient dimension
    gradient = np.linspace(min_sig, max_sig, field_shape[-1])
    
    #Initialise shape for broadcasting 
    shape = [1] * dim
    shape[-1] = field_shape[-1]
    
    #reshape the gradient for broadcasting
    gradient = gradient.reshape(shape)
    
    #generating the signal
    signal = np.zeros(field_shape)
    signal += gradient
    
    return(signal)

def gaussian_noise_field(field_shape, n_subject, noise_fwhm, noise_scale):
    
    smoother = fwhm_to_sigma(noise_fwhm)
    
    instance = []
    
    for i in range(n_subject):
        
        epsilon = np.random.randn(*field_shape)
        
        if smoother > 0:
            epsilon = ndimage.gaussian_filter(epsilon, smoother)
        
        epsilon /= epsilon.std(ddof=1)
        
        epsilon *= noise_scale
        
        instance.append(epsilon)
    
    instance = np.stack(instance)
    return instance

def compute_boundary(mask):

    """
    Compute the boundary of a boolean mask in either 2D or 3D
    """

    bdry = np.zeros_like(mask, dtype=bool)

    s0 = []  # outside
    s1 = []  # inside

    if mask.ndim == 2:

        up = mask[1:, :] & ~mask[:-1, :]
        bdry[1:, :] |= up
        coord = np.argwhere(up)
        s0.append(coord)
        s1.append(coord + [1, 0])

        down = ~mask[1:, :] & mask[:-1, :]
        bdry[:-1, :] |= down
        coord = np.argwhere(down)
        s0.append(coord + [1, 0])
        s1.append(coord)

        left = mask[:, 1:] & ~mask[:, :-1]
        bdry[:, 1:] |= left
        coord = np.argwhere(left)
        s0.append(coord)
        s1.append(coord + [0, 1])

        right = ~mask[:, 1:] & mask[:, :-1]
        bdry[:, :-1] |= right
        coord = np.argwhere(right)
        s0.append(coord + [0, 1])
        s1.append(coord)

    elif mask.ndim == 3:

        # axis 0 crossings
        up = mask[1:, :, :] & ~mask[:-1, :, :]
        bdry[1:, :, :] |= up
        coord = np.argwhere(up)
        s0.append(coord)
        s1.append(coord + [1, 0, 0])

        down = ~mask[1:, :, :] & mask[:-1, :, :]
        bdry[:-1, :, :] |= down
        coord = np.argwhere(down)
        s0.append(coord + [1, 0, 0])
        s1.append(coord)

        # axis 1 crossings
        left = mask[:, 1:, :] & ~mask[:, :-1, :]
        bdry[:, 1:, :] |= left
        coord = np.argwhere(left)
        s0.append(coord)
        s1.append(coord + [0, 1, 0])

        right = ~mask[:, 1:, :] & mask[:, :-1, :]
        bdry[:, :-1, :] |= right
        coord = np.argwhere(right)
        s0.append(coord + [0, 1, 0])
        s1.append(coord)

        # axis 2 crossings
        front = mask[:, :, 1:] & ~mask[:, :, :-1]
        bdry[:, :, 1:] |= front
        coord = np.argwhere(front)
        s0.append(coord)
        s1.append(coord + [0, 0, 1])

        back = ~mask[:, :, 1:] & mask[:, :, :-1]
        bdry[:, :, :-1] |= back
        coord = np.argwhere(back)
        s0.append(coord + [0, 0, 1])
        s1.append(coord)

    else:
        raise ValueError("compute_boundary only supports 2D or 3D masks.")

    if len(s0) == 0:
        s0 = np.empty((0, mask.ndim), dtype=int)
        s1 = np.empty((0, mask.ndim), dtype=int)
    else:
        s0 = np.vstack(s0)
        s1 = np.vstack(s1)

    return bdry, s1, s0
    
def simulate_instances(signal, n_subject, noise_fwhm, noise_scale):
    
    field_shape = signal.shape
    noise = gaussian_noise_field(field_shape, n_subject, 
                                 noise_fwhm, noise_scale)
    
    instances = signal[None] + noise

    return(instances)

def contour_interpolation(field, threshold):
    
    mask = field > threshold
    
    bdry, s1, s0 = compute_boundary(mask)
    
    dim = np.ndim(field)
    
    
    if s0.size == 0 or s1.size == 0:
     return (
         np.empty((0, dim), dtype=float),
         bdry,
         np.empty((0, dim), dtype=int),
         np.empty((0, dim), dtype=int),
         np.empty((0,), dtype=float),
         np.empty((0,), dtype=float),
     )
    
    if dim == 2:
        b0 = field[s0[:, 0], s0[:, 1]]
        b1 = field[s1[:, 0], s1[:, 1]]

    elif dim == 3:
        b0 = field[s0[:, 0], s0[:, 1], s0[:, 2]]
        b1 = field[s1[:, 0], s1[:, 1], s1[:, 2]]
    
    else:
        raise ValueError("Only 2D and 3D fields are supported.")
    
    m1 = (b1 - threshold)/(b1-b0)
    m2 = 1-m1
    
    points = m1[:, None] * s0 + m2[:, None] * s1
    
    return(points, bdry, s0, s1, m1, m2)

def rademacher(n):
    """
    generate n sample of 1 or -1 with probability 1/2 for each outcome
    """
    outcome = np.random.choice([-1,1], size = n, replace = True)
    return(outcome)

def directed_hausdorff_distance(a, b):

    if len(a) == 0 or len(b) == 0:
        return np.inf

    return directed_hausdorff(a, b)[0]


def symmetric_hausdorff(a, b):

    if len(a) == 0 or len(b) == 0:
        return np.inf

    d_ab = directed_hausdorff(a, b)[0]
    d_ba = directed_hausdorff(b, a)[0]

    return max(d_ab, d_ba)





    

    
    
    
