# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:01:20 2025

@author: railt
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv

def rademacher(n):
    """
    generate n sample of 1 or -1 with probability 1/2 for each outcome
    """
    outcome = np.random.choice([-1,1], size = n, replace = True)
    return(outcome)

def compute_boundary(mask):
    """
    

    Parameters
    ----------
    mask : 2d array of boolean value
        array of 2d values of siginificant region of interest

    Returns
    -------
    boundary points, s1 = inside boundary, s0 = outside boundary

    """
    bdry = np.zeros_like(mask, dtype = bool)

    s0=[] #outside
    s1=[] #inside
    ver_s0 =[]
    ver_s1 =[]
    hor_s0 =[]
    hor_s1 = []
    
    up = (mask[1:,:] & ~mask[:-1,:])
    bdry[1:,:] |= up
    coord = np.argwhere(up)
    s0.append(coord )
    s1.append(coord + [1,0])
    ver_s0.append(coord)
    ver_s1.append(coord + [1,0])
    
    down = (~mask[1:,:] & mask[:-1,:]) 
    bdry[:-1,:] |= down
    coord = np.argwhere(down)
    s0.append(coord + [1,0])
    s1.append(coord)
    ver_s0.append(coord + [1,0])
    ver_s1.append(coord)

    left = (mask[:,1:] & ~mask[:,:-1])
    bdry[:,1:] |= left 
    coord = np.argwhere(left)
    s0.append(coord)
    s1.append(coord + [0,1])
    hor_s0.append(coord)
    hor_s1.append(coord + [0,1])
    
    right = (~mask[:,1:] & mask[:,:-1])
    bdry[:,:-1] |= (~mask[:,1:] & mask[:,:-1])
    coord=np.argwhere(right)
    s0.append(coord + [0,1])
    s1.append(coord)
    hor_s0.append(coord + [0,1])
    hor_s1.append(coord)

    s0 = np.concatenate(s0)
    s1 = np.concatenate(s1)
    ver_s0 = np.concatenate(ver_s0)
    ver_s1 = np.concatenate(ver_s1)
    hor_s0 = np.concatenate(hor_s0)
    hor_s1 = np.concatenate(hor_s1)
    return(bdry, s1, s0, ver_s1, ver_s0, hor_s1, hor_s0)

def generate_circle_100grid(radius, magnitude):
    """
    
    """
    
    center = np.array([50,50])
    x = np.meshgrid(np.arange(100),np.arange(100))
    circle_mask =  (x[0] - center[0])**2 + (x[1] - center[1])**2 <= radius **2
    
    #generating a circle
    circle = np.zeros((100,100))
    signif_point = np.where(circle_mask)
    circle[signif_point] = magnitude
    return(circle, signif_point)
def generate_cross():
    """
    generate a + symbol
    """
    boi = np.zeros((100,100))
    boi[40:61,50] = 6
    boi[50,40:61] = 6
    return(boi)

def generate_gradient(minsig,maxsig):
    """
    generate a gradient signal     
    """

    gradient = np.linspace(minsig,maxsig,100)
    gradient = np.tile(gradient, (100,1))
    return(gradient)

def boundary_dilation(bdry):
    
    #initialise some values
    bdry_dil = np.zeros_like(bdry, dtype = bool)
    #upward dilation
    bdry_dil[:-1,:] |= bdry[1:,:]

    #downward dilation
    bdry_dil[1:,:] |= bdry[:-1,:]
    
    #left
    bdry_dil[:,:-1] |= bdry[:,1:]
    
    #right 
    bdry_dil[:,1:] |= bdry[:,:-1]

    #include original boundary
    bdry_dil |= bdry
    
    return(bdry_dil)

def hausdorff(mask1,mask2):
    
    #initialise some distance
    subset1_yet = False
    subset2_yet = False
    epsilon1 = 0
    epsilon2 = 0
    bdry1 = compute_boundary(mask1)[0]
    bdry2 = compute_boundary(mask2)[0]
    
    #initialise dilation
    bdry1_dil = bdry1
    bdry2_dil = bdry2
    
    while not subset1_yet:
        #check subset
        if np.all(bdry1_dil >= bdry2):
            subset1_yet = True 
        else:
            bdry1_dil = boundary_dilation(bdry1_dil)
            epsilon1 += 1
    
    while not subset2_yet:
        #check subset
        if np.all(bdry2_dil >= bdry1):
            subset2_yet = True
        else:
            bdry2_dil = boundary_dilation(bdry2_dil)
            epsilon2 += 1
    
    haus = max(epsilon1, epsilon2)
    return(haus)

def numerical_test(samples, boot_rep, alpha, fwhm, threshold, signal_type, repeat):
    
    if signal_type =='circle':
        signal = generate_circle_100grid(30,3)[0]
        signal = ndimage.gaussian_filter(signal, 3/(2*np.sqrt(2*np.log(2))))
    elif signal_type =='grad':
        signal = generate_gradient(0,3)
        signal = ndimage.gaussian_filter(signal, 3/(2*np.sqrt(2*np.log(2))))
    else:
        return(print('error'))
    
    success = 0
    
    for i in range(repeat):
        print(i)

        true_mask = signal > threshold
        bdry_list = compute_boundary(true_mask)
        true_bdry, true_s1, true_s0 = bdry_list[0], bdry_list[1], bdry_list[2]
        
        true_b_s0 = signal[true_s0[:,0], true_s0[:,1]]
        true_b_s1 = signal[true_s1[:,0], true_s1[:,1]]
        
        
        true_diff = true_b_s1 - true_b_s0
        
        true_m1 = (true_b_s1 - threshold)/true_diff
        true_m2 = (threshold - true_b_s0)/true_diff
        
        
        
        #generating instances  of smoothed circle + smoothed noise
        instances = []
        #eppy = []
        for i in range(samples):
            epsilon = ndimage.gaussian_filter(np.random.randn(100,100), fwhm/(2*np.sqrt(2*np.log(2))))
            epsilon /= epsilon.std() #renormalise epsilon
            #eppy.append(epsilon)
            instances.append(signal + epsilon) #3fwhm
            
            
        #using the fact that Linear regression consist only of intercept. 
        #The Intercept's coefficient is just the mean value we have that can compute beta easily
        instances = np.stack(instances)
        coefficient = np.mean(instances, axis = 0)
        
        
        
        #mask for the significant regression coefficient values
        mask = coefficient > threshold
        coef_bdry_list =compute_boundary(mask)
        bdry, s1, s0 = coef_bdry_list[0], coef_bdry_list[1], coef_bdry_list[2]
        
        #register coefficients values at the inside boundary s0 and its neighbour outside boundary s1
        b_s0 = coefficient[s0[:,0], s0[:,1]]
        b_s1 = coefficient[s1[:,0], s1[:,1]]
        
        diff = np.abs(b_s1 - b_s0)
        
        #weight based on distances the regression coeffcient has from threshold (assume linear distance based on c)
        m1 = (b_s1 - threshold)/diff
        m2 = (threshold - b_s0)/diff
        
        
        
        epsilons  = (instances - coefficient)
        eps_std = epsilons.std(axis=0, ddof=1)
        epsilons = (epsilons)/eps_std
        
        epsilon_star = m1 * epsilons[:,s0[:,0], s0[:,1]] + m2 * epsilons[:,s1[:,0], s1[:,1]]
             
        #computing G*
        number_subject, number_point = np.shape(epsilon_star)


        #computing G the way alex code it (less efficient)
        G = []
        for i in range(boot_rep):
            rad = rademacher(number_subject)[:, None, None]
            boot_residual = rad * epsilons
            boot_std = boot_residual.std(axis=0, ddof=1)
            t_field = boot_residual.sum(axis=0) / (boot_std * np.sqrt(number_subject))
            t_star = m1 * t_field[s0[:,0], s0[:,1]] + m2 * t_field[s1[:,0], s1[:,1]]
            G.append(np.max(np.abs(t_star)))
        

        k = np.quantile(G, 1-alpha)
        
        #weight
        vw = 1/np.sqrt(number_subject)
        
        #compute upper_lower region
        upper_coeff = coefficient - k * eps_std * vw
        lower_coeff = coefficient + k * eps_std * vw
         
        upper_mask = upper_coeff >= threshold
        lower_mask = lower_coeff >= threshold
        
        upper_region = np.argwhere(upper_mask) #AC+
        lower_region = np.argwhere(lower_mask)  #AC-
        
        
        
        #value of upper and lower region at true boundaries
        upper_s0 = upper_coeff[true_s0[:,0],true_s0[:,1]]
        upper_s1 = upper_coeff[true_s1[:,0],true_s1[:,1]]
        lower_s0 = lower_coeff[true_s0[:,0],true_s0[:,1]]
        lower_s1 = lower_coeff[true_s1[:,0],true_s1[:,1]]
        
        upper_boundary = true_m1 * (upper_s0) + true_m2 * (upper_s1) 
        lower_boundary = true_m1 * (lower_s0) + true_m2 * (lower_s1) 
        
        #test if the condition failed
        upper_test = upper_boundary >= threshold # if s* in upper then violate
        lower_test = lower_boundary < threshold#if s* not in lower then violated
        
        test = not ((np.any(upper_test) or np.any(lower_test)))
            
        print(test)
        
        if test:
            success += 1
    print(success)
    print(repeat)
    success_rate = success/repeat
    return(success_rate)

def interpolate1(coefficient, thr):
    
    mask = coefficient > thr
    bdry_list = compute_boundary(mask)
    
    bdry, s1, s0, ver_s1, ver_s0, hor_s1, hor_s0 = bdry_list
    
    b_s0_ver = coefficient[ver_s0[:,0], ver_s0[:,1]]
    b_s1_ver = coefficient[ver_s1[:,0], ver_s1[:,1]]
    
    b_s0_hor = coefficient[hor_s0[:,0], hor_s0[:,1]]
    b_s1_hor = coefficient[hor_s1[:,0], hor_s1[:,1]]
    
    diff_ver = np.abs(b_s0_ver - b_s1_ver)
    diff_hor = np.abs(b_s0_hor - b_s1_hor)
    
    m1_ver = (b_s1_ver - thr)/diff_ver
    m2_ver = (thr - b_s0_ver)/diff_ver
    m1_hor = (b_s1_hor - thr)/diff_hor
    m2_hor = (thr - b_s0_hor)/diff_hor
    
    s_star_ver = np.concatenate(([m1_ver * ver_s0[:,0] + m2_ver * ver_s1[:,0]]))
    s_star_ver = np.stack((s_star_ver, ver_s1[:,1]), axis=1)
    s_star_hor = np.concatenate(([m1_hor * hor_s0[:,1] + m2_hor * hor_s1[:,1]]))
    s_star_hor = np.stack((hor_s1[:,0], s_star_hor), axis=1)
    
    yolo = np.vstack((s_star_ver, s_star_hor))
    
    return(yolo)