"""
Example script showing use of mcphysics.data to load *.CHN files 
and subsequent manipulation of spinmob databoxes containing the data.

You need the companion .chn file, "Example_data.chn" to run this script.

Author: Brandon Ruffolo
"""
import mcphysics.data as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = m.load_chn("Long_Am_Run.Chn")


# Print the hkeys (metadata fields)
print("hkeys: ")
print(data.hkeys)
print()


# Print the start time of the run
print("Start time: ")
print(data.h('start_time'))
print()

# Print the run description (if you enetered one!)
print("Run description:")
print(data.h('description')) 
print()

# Print the ckeys (data fields)
print("ckeys: ")
print(data.ckeys)
print()

# Print the channel data (array spanning from 0 to 2047, since the MCA is 11-bit)
print("Channel data: ")
print(data['Channel'])
print()


plt.figure()
plt.plot(data['Channel'],data['Counts'])
plt.xlabel("Channel Number", fontsize=14); plt.ylabel("Counts", fontsize=14)
plt.show()
plt.show(block=True)

def gaussian_bg(x, A, x0, sigma, B):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B

# Find peak
max_index = data['Counts'].argmax()
max_count = data['Counts'][max_index]


# Fit window
fit_min = max_index - 8
fit_max = max_index + 8

mask = (data['Channel'] >= fit_min) & (data['Channel'] <= fit_max)

x_fit = data['Channel'][mask]
y_fit = data['Counts'][mask]

# Initial guesses
B0 = np.median(y_fit)
sigma0 = 7
A0 = max_count - B0

p0 = [A0, max_index, sigma0, B0]

params, cov = curve_fit(gaussian_bg, x_fit, y_fit, p0=p0)

A, x0, sigma, B = params

print(f"Peak channel (fit): {x0:.2f}")
print(f"Sigma: {sigma:.2f}")
print(f"FWHM: {2.355*sigma:.2f} channels")

# Plot
x_fine = np.linspace(fit_min, fit_max, 1000)
y_fine = gaussian_bg(x_fine, *params)

plt.figure()
plt.plot(x_fit, y_fit, 'o', label="Data")
plt.plot(x_fine, y_fine, '-', label="Gaussian fit")
plt.xlabel("Channel Number")
plt.ylabel("Counts")
plt.legend()
plt.show()
plt.show(block=True)

residuals = y_fit - gaussian_bg(x_fit, *params)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x_fit, y_fit, 'o', label='Data')
plt.plot(x_fine, y_fine, '-', label='Fit')
plt.legend()

plt.subplot(2,1,2)
plt.plot(x_fit, residuals, 'o')
plt.axhline(0, color='k')
plt.ylabel("Residuals")
plt.xlabel("Channel")

plt.show()
plt.show(block=True)