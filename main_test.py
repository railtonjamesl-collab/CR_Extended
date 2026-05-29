# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:49:04 2026

@author: railt
"""

import utility as util
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.spatial.distance import directed_hausdorff

#initialising parameters
field_shape = (100,100)
n_subject = 100
threshold = 2
boot_rep = 100

signal_type = 'circle'
signal_fwhm = 3
noise_fwhm = 3
noise_scale = 1

radius = 30
magnitude = 3
minsig = 0
maxsig = 3
alhpa = 0.1


#generate signal
if signal_type == 'circle':
    signal_field = util.ball_signal(field_shape, radius, magnitude)

if signal_type == 'gradient':
    signal_field = util.gradient_signal(field_shape, minsig, maxsig)
    
#smoothing the signal
signal_sigma = util.fwhm_to_sigma(signal_fwhm)
signal_field = ndimage.gaussian_filter(signal_field, signal_sigma)


#generate and add the smoothed noise to the signal field
noise_field = util.gaussian_noise_field(field_shape, n_subject, noise_fwhm, noise_scale)
signal_field = signal_field + noise_field









