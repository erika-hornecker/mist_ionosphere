#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:20:16 2021

@author: erika_000
"""

import numpy as np
import math


import iri2016 as ion
from iri2016 import altitude
from iri2016 import latitude

from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#from astropy_healpix import HEALPix
#from astropy.coordinates import Galactic
#from astropy import units as u

import pymap3d as pm

import time

import os
import h5py

#start = time.time()


#Freq in Hz
def run_all (filename, lat0, lon0, h0, date_time, freq, f_lower, f_upper, num_f_layers, num_d_layers, col_freq, save_y_n, save_name):
    
    """
    #Coordinates of the instrument
    lat0= -26.7148
    lon0= 116.6044
    h0= 0
    global date_time
    date_time = '2012-06-28 22:00'
    """
    
    d_bottom = 60000
    d_top = 90000
    
    h_D = d_bottom + (d_top - d_bottom)/2
    delta_hD = (d_top - d_bottom)

    
    global el, az, n_pts
    el, az, n_pts = get_pts(filename)
    
    d_srange = d_srange_pts(num_d_layers, el)
    d_e_density, d_e_temp = get_d_params(lat0, lon0, h0, date_time, el, az, d_srange, num_d_layers)
    attenuation = get_attenuation(d_e_density, el, freq, col_freq, h_D, delta_hD, num_d_layers)
    
    f_e_density, phis, delta_theta, ref_indices = get_refraction(lat0, lon0, h0, date_time, freq, f_lower, f_upper, num_f_layers)
    
    d_avg_temp = np.zeros(n_pts)
    
    for i in range(n_pts):
        temp_data = []
        for j in range(num_d_layers):
            if (d_e_temp[i][j] > 0):
                temp_data.append(d_e_temp[i][j])
        d_avg_temp[i] = np.average(temp_data)
    
    
    dt = datetime.fromisoformat(date_time)

     # Save
     # ----------------------
    
    output_file_name_hdf5 = str(lat0)+'_'+str(lon0)+'_'+str(h0)+'_'+dt.strftime("%m-%d-%Y_%H-%M")
    if save_y_n == 'y':

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        with h5py.File(save_name + '_' + output_file_name_hdf5, 'w') as hf:

            hf.create_dataset('f_lower', data = f_lower)
            hf.create_dataset('f_upper', data = f_upper)
            hf.create_dataset('num_d_layers', data = num_d_layers)
            hf.create_dataset('num_f_layers', data = num_f_layers)
            hf.create_dataset('collision_freq', data = col_freq)
            
            hf.create_dataset('azimuth', data = az)
            hf.create_dataset('elevation', data = el)
            hf.create_dataset('frequency', data = freq)

            hf.create_dataset('f_e_density', data = f_e_density)
            hf.create_dataset('layers_ref_angles', data = phis)
            hf.create_dataset('layers_ref_indices', data = ref_indices)
            
            hf.create_dataset('d_e_density', data = d_e_density)
            hf.create_dataset('d_e_temp', data = d_e_temp)
            hf.create_dataset('attenuation', data = attenuation)
            hf.create_dataset('d_avg_temp', data = d_avg_temp)
   
            hf.close()
    
    return attenuation, d_e_density, f_e_density, phis, delta_theta, ref_indices


def create_pts (num_pts):
    
    file = open('az_el_pts.txt', 'w')

    
    #Azimuth in degrees
    #az= np.random.rand(num_pts)*360
    az_vals = np.linspace(0, 360, num_pts)
    az = np.repeat(az_vals, num_pts)
    #az = np.full(shape=num_pts, fill_value=0)

    #Elevation in degrees
    #el= np.random.rand(num_pts)*90
    #el = np.linspace(0, 90, num_pts)
    el_vals = np.linspace(0, 90, num_pts)
    el = np.tile(el_vals, num_pts)
    
    #saves both arrays in succession 

    np.savetxt(file, (el, az))
    
    file.close()


def get_pts (filename):
    
    file = np.loadtxt(filename)

    el = file[0]
    az = file[1]
    
    n_pts = len(el)
    
    return el, az, n_pts


#theta: zenith angle in rad
#alt: alititde of point
#distances in meters
def range_val (theta, alt):
        RE = 6378000
        r = -RE*np.cos(theta)+np.sqrt(RE**2*np.cos(theta)**2+(alt**2)+2*alt*RE)
        return r


#FUNCTION: to get the srange values to each layer for each elevation 
# Heights in meters
def d_srange_pts (resolution, el):
    #Set resolution in range (i.e. number of range points between layer limits)
    srange_resol = resolution

    d_bottom = 60000
    d_top = 90000
    d_heights = np.linspace(d_bottom, d_top, srange_resol)

    d_srange = np.empty([len(el), srange_resol])

        
    for i in range(len(el)):
        for j in range(srange_resol):
            d_srange[i][j] = range_val((90-el[i])*np.pi/180, d_heights[j])
    
    return d_srange

    
    
def get_d_params (lat0, lon0, h0, date_time, el, az, d_srange, srange_resol):

    global d_obs_h

    d_obs_lat = np.zeros([n_pts, srange_resol])
    d_obs_lon = np.zeros([n_pts, srange_resol])
    d_obs_h = np.zeros([n_pts, srange_resol])
    
    #D-layer electron density and temperature 
    d_e_density = np.zeros([n_pts, srange_resol])
    d_e_temp = np.zeros([n_pts, srange_resol])
    
    #The latitude, longitude, and height of the object, as seen from the surface of the Earth
    for i in range (n_pts):
        for j in range(srange_resol):
            d_obs_lat[i][j], d_obs_lon[i][j], d_obs_h[i][j] = pm.aer2geodetic(az[i], el[i], d_srange[i][j], lat0, lon0, h0)


            d_alt_range = [d_obs_h[i][j]/1000, d_obs_h[i][j]/1000, 1]
            d_alt_prof = ion.IRI(date_time, d_alt_range , d_obs_lat[i][j], d_obs_lon[i][j])
            
            if (d_alt_prof.ne.data > 0):
                d_e_density[i][j]= d_alt_prof.ne.data
            else: 
                d_e_density[i][j]= 0
                
            d_e_temp[i][j]= d_alt_prof.Te.data
            
            print("D-Layer: point ", i, "/", n_pts, " and layer ", j, "/", srange_resol)
    
    
    #ion_params_time = time.time()
    
    #print("Runtime for ", n_pts, " sky points and ", srange_resol, " layers is: ", (ion_params_time-start), " seconds.")
    
    return d_e_density, d_e_temp


def trop (theta):
    a = 16709
    b= -19066.21
    c= 5396.33
    return 1.0/(a+b*theta+c*(theta)**2)

#Get plasma frequency [Hz] from electron density 
def nu_p (n_e):
    e = 1.60217662e-19
    m_e = 9.10938356e-31
    epsilon0 = 8.85418782e-12
    return 1/(2*np.pi)*np.sqrt((n_e*e**2)/(m_e*epsilon0))

#Refractive index of F-layer from electron density
def n_f (n_e, nu):
    return (1-(nu_p(n_e)/nu)**2)**(0.5)


def col_nicolet (height):
    
    a = -0.16184565
    b = 28.02068763
    
    col_freqs = np.exp(a*height+b)
    
    return col_freqs


def col_setty (heights):
    
    a = -0.16018896
    b = 26.14939429
    
    col_freqs = np.exp(a*heights+b)
    
    return col_freqs


def col_avg_model (heights):
    
    cols1 = col_nicolet(heights)
    cols2 = col_setty(heights)
    
    avg_cols = (cols1+cols2)/2
    
    return avg_cols

"""
nu: frequency [Hz] 
theta: elevation angle [°] 
n_e: array of electron densities in given line of sight [m^-3]
h: array of heights in given line of sight [m]
"""


"""
Attenuation factor:
- inputs are the frequency of the signal [Hz], angle [rad], altitude of the D-layer midpoint [km],
thickness of the D-layer [km], plasma frequency [Hz], and electron collision frequency [Hz]
- output is the attenuation factor between 0 (total attenuation) and 1 (no attenuation)
"""

def d_atten (nu, theta, h_D, delta_hD, nu_p, nu_c):
    R_E = 6371000
    c = 2.99792458e8
    delta_s = delta_hD*(1+h_D/R_E)*(np.cos(theta)**2+2*h_D/R_E)**(-0.5)
    f = np.exp(-(2*np.pi*nu_p**2*nu_c*delta_s)/(c*(nu_c**2+nu**2)))
    
    return f

"""
Brightness temperature: the temp seen by the antenna inside the ionosphere T_sky in relation to the
"actual" temp outside of the ionosphere T_out, with T_e the D-layer electron temperature
"""

def temp (T_out, f_att, T_e):
    T_sky = f_att*T_out+(1-f_att)*T_e
    return T_sky



def get_attenuation (d_e_density, el, freq, col_freq, h_D, delta_hD, n_d_layers):
    
    d_attenuation = np.zeros(n_pts)
    
    #If 0, then use default value for collision frequency (10MHz)
    if (col_freq == 0):
        nu_c = 10e6
        d_avg_density = np.zeros(n_pts)
                
        for i in range(n_pts):
            
            d_avg_density[i] = np.average(d_e_density[i])
            plasma_freq = nu_p(d_avg_density[i])
            d_attenuation[i] = d_atten(freq, (90-el[i])*np.pi/180, h_D, delta_hD, plasma_freq, nu_c)
    
    
    #If 1, then use height dependent collision frequencies from Nicolet model
    elif (col_freq == 1):
        
        for i in range(n_pts):
            nu_c = col_nicolet(d_obs_h[i]/1000)
            d_attenuation_temp = np.zeros(n_d_layers)
            
            for j in range(n_d_layers):
                plasma_freq = nu_p(d_e_density[i][j])
                d_attenuation_temp[j] = d_atten(freq, (90-el[i])*np.pi/180, h_D, delta_hD, plasma_freq, nu_c[j])
            
            d_attenuation[i] = np.average(d_attenuation_temp)
        
    #If 2, then use height dependent collision frequencies from Setty model
    elif (col_freq == 2):
        
        for i in range(n_pts):
            nu_c = col_setty(d_obs_h[i]/1000)
            d_attenuation_temp = np.zeros(n_d_layers)
            
            for j in range(n_d_layers):
                plasma_freq = nu_p(d_e_density[i][j])
                d_attenuation_temp[j] = d_atten(freq, (90-el[i])*np.pi/180, h_D, delta_hD, plasma_freq, nu_c[j])
            
            d_attenuation[i] = np.average(d_attenuation_temp)
    
        
    #If 3, then use average of both height dependent models
    elif (col_freq == 3):
       
        for i in range(n_pts):
            nu_c = col_avg_model(d_obs_h[i]/1000)
            d_attenuation_temp = np.zeros(n_d_layers)
            
            for j in range(n_d_layers):
                plasma_freq = nu_p(d_e_density[i][j])
                d_attenuation_temp[j] = d_atten(freq, (90-el[i])*np.pi/180, h_D, delta_hD, plasma_freq, nu_c[j])
            
            d_attenuation[i] = np.average(d_attenuation_temp)
        
    else: 
        nu_c = col_freq
        d_avg_density = np.zeros(n_pts)
        
        for i in range(n_pts):
        
            d_avg_density[i] = np.average(d_e_density[i])
            plasma_freq = nu_p(d_avg_density[i])
            d_attenuation[i] = d_atten(freq, (90-el[i])*np.pi/180, h_D, delta_hD, plasma_freq, nu_c)
        
        
    return d_attenuation



#Input angle in rad
def snell (n1, n2, phi_i):
    phi_f = np.arcsin(n1/n2*np.sin(phi_i))
    return phi_f



#FRQ IN HZ 
def get_refraction (lat_i, lon_i, h_i, date_time, freq, f_lower, f_upper, n_f_layers):
    
    R_E = 6371000
    f_heights = np.linspace(f_lower, f_upper, n_f_layers)
    
    ns = np.zeros([n_pts, n_f_layers])
    f_e_density = np.zeros([n_pts, n_f_layers])
    phis = np.zeros([n_pts, n_f_layers])
    delta_phi = np.zeros(n_pts)


    for i in range(n_pts):
        
        delta_phi_temp = 0
        
        r1 = range_val((90-el[i])*np.pi/180, f_heights[0])
        
        #Get geodetic coordinates of point 
        lat1, lon1, h1 = pm.aer2geodetic(az[i], el[i], r1, lat_i, lon_i, h_i)
        
        #The sides of the 1st triangle
        d1 = R_E + h_i
        d2 = R_E + h1
        
        #The incoming angle at the 1st interface using law of cosines [rad]
        phi1 = np.arccos((r1**2+d2**2-d1**2)/(2*r1*d2))
    
        
        #Refraction index of air
        n1 = 1
        
        
        #Get IRI info of point
        f_alt_range = [h1/1000, h1/1000, 1]
        f_alt_prof = ion.IRI(date_time, f_alt_range , lat_i, lon_i)
        n_e = f_alt_prof.ne.data[0]
        f_e_density[i][0] = n_e
        
        #Refraction index of 1st point
        n2 = n_f(n_e, freq)
        ns[i][0] = n2
        
        #The outgoing angle at the 1st interface using Snell's law
        phi2 = snell(n1, n2, phi1)
        phis[i][0] = phi2
        delta_phi_temp += (phi2-phi1)
        
        el1 = el[i] - (phi2-phi1)
        
        ##################################
        
        for j in range(n_f_layers-1):
            
            print('F-Layer: point ', i, '/', n_pts, ' and layer ', j+1, '/', n_f_layers)
            
            #The internal angle of the next triangle
            int_angle = np.pi - phi2
            
            h3 = f_heights[j+1]
            
            d3 = R_E + h3
            
            #The incoming angle at the 2nd interface using law of sines [rad]
            phi3 = np.arcsin(np.sin(int_angle)*d2/d3)
            
            #Getting r2 using law of cosines
            r2 = d2*np.cos(int_angle)+np.sqrt(d3**2-d2**2*np.sin(int_angle)**2)
            
            #Get geodetic coordinates of point 
            lat2, lon2, h2 = pm.aer2geodetic(az[i], el1, r2, lat1, lon1, h1)
            
            
            #Get IRI info of 2nd point
            f_alt_range = [h2/1000, h2/1000, 1]
            f_alt_prof = ion.IRI(date_time, f_alt_range , lat2, lon2)
            n_e2 = f_alt_prof.ne.data[0]
            f_e_density[i][j+1] = n_e2
            
            #Refractive indices
            n3 = n_f(n_e2, freq)
            ns[i][j+1] = n3
            
            #If this is the last point then use refractive index of vacuum
            if (j == n_f_layers-2):
                n3 = 1
            
            #The outgoing angle at the 2nd interface using Snell's law
            phi4 = snell(n2, n3, phi3)
            phis[i][j+1] = phi4
            delta_phi_temp += (phi4-phi3)
            
            #Update variables for new interface
            el1 = el1 - (phi4-phi3)
            phi2 = phi4
            
            lat1 = lat2
            lon1 = lon2
            h1 = h2
            
            n2 = n3
            d2 = d3
            
        delta_phi[i] = delta_phi_temp
    
        
    
    return f_e_density, phis, delta_phi, ns
    

def get_direct_f_e_dens (filename, lat0, lon0, h0, date_time, f_lower, f_upper, num_f_layers):
    
    el, az, n_pts = get_pts(filename)

    f_heights = np.linspace(f_lower, f_upper, num_f_layers)
    f_srange = np.empty([len(el), num_f_layers])
    
    f_obs_lat = np.zeros([n_pts, num_f_layers])
    f_obs_lon = np.zeros([n_pts, num_f_layers])
    f_obs_h = np.zeros([n_pts, num_f_layers])
    
    f_e_density = np.zeros([n_pts, num_f_layers])
        
    for i in range(n_pts):
        for j in range(num_f_layers):
            
            f_srange[i][j] = range_val((90-el[i])*np.pi/180, f_heights[j])

            f_obs_lat[i][j], f_obs_lon[i][j], f_obs_h[i][j] = pm.aer2geodetic(az[i], el[i], f_srange[i][j], lat0, lon0, h0)
            
            f_alt_range = [f_obs_h[i][j]/1000, f_obs_h[i][j]/1000, 1]
            f_alt_prof = ion.IRI(date_time, f_alt_range , f_obs_lat[i][j], f_obs_lon[i][j])
          
            f_e_density[i][j]= f_alt_prof.ne.data
            
            print("F-Layer: point ", i, "/", n_pts, " and layer ", j, "/", num_f_layers)
            

    return f_e_density
    
def polar_plot(el, az, data, n_rows, title, label):
    
    """
    #THIS IS TO GET AVERAGES FOR A 2D DATASET (ex:for e_density plots):
    avg_data = np.zeros(len(data))
    for i in range(len(data)):
        avg_data[i] = np.average(data[i])
        
    az_2 = np.zeros(len(az))
    zenith = np.zeros(len(el))

    
    for i in range(len(el)):
        az_2[i] = az[i]*np.pi/180
        zenith[i] = 90-el[i]
        
    zen = np.split(zenith, n_rows)    
    azi = np.split(az_2, n_rows)
    data2 = np.split(avg_data, n_rows)
    """
    
    
    #THIS IS WHEN USING A 1D DATASET
    az_2 = np.zeros(len(az))
    zenith = np.zeros(len(el))
    
    for i in range(len(el)):
        az_2[i] = az[i]*np.pi/180
        zenith[i] = 90-el[i]
        
    zen = np.split(zenith, n_rows)    
    azi = np.split(az_2, n_rows)
    data2 = np.split(data, n_rows)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='polar')
    img = ax.pcolormesh(azi, zen, data2, cmap='viridis')
    ax.set_rticks([90, 60, 30, 0])
    ax.set_theta_zero_location("S")
    #plt.colorbar(img)
    plt.colorbar(img).set_label(r''+label)
    plt.title(label=title)
    plt.show()
       