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
n_subject = 50
threshold = 2
boot_rep = 1000

signal_type = 'circle'
signal_fwhm = 3
noise_fwhm = 3
noise_scale = 1

radius = 30
magnitude = 3
minsig = 0
maxsig = 3
alpha = 0.1


#generate signal
if signal_type == 'circle':
    signal_field = util.ball_signal(field_shape, radius, magnitude)

if signal_type == 'gradient':
    signal_field = util.gradient_signal(field_shape, minsig, maxsig)
    
#smoothing the signal
signal_sigma = util.fwhm_to_sigma(signal_fwhm)
signal_field = ndimage.gaussian_filter(signal_field, signal_sigma)

#generate instances by adding the signal field to a smoothed gaussian
noise_fields = util.gaussian_noise_field(field_shape, n_subject, noise_fwhm, noise_scale)
instances = signal_field + noise_fields

#compute ols estimator
coefficient = np.mean(instances, axis = 0)

#Bowring method's Bootstraping
supremum_values = util.Bowring_bootstrap(instances, coefficient, boot_rep, threshold)
k = np.quantile(supremum_values, 1-alpha)


result = util.contour_interpolation(coefficient, threshold)
    















