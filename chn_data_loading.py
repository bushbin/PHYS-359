import mcphysics.data as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = m.load_chn("Long_Am_Run.Chn")
uncertainties = np.sqrt(data['Counts'])

print(data.h('live_time'))

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

print(np.sqrt(cov[1,1]))

A, x0, sigma, B = params

print(f"Peak channel (fit): {x0:.2f}")
print(f"Sigma: {sigma:.2f}")
print(f"FWHM: {2.355*sigma:.2f} channels")

# Plot
x_fine = np.linspace(fit_min, fit_max, 1000)
y_fine = gaussian_bg(x_fine, *params)

plt.figure()
plt.plot(x_fit, y_fit, 'o', label="Data", color='red' )
plt.plot(x_fine, y_fine, '-', label="Gaussian fit", color='blue')
plt.errorbar(x_fit, y_fit, np.sqrt(y_fit), fmt='o', capsize=4, color='red')
plt.xlabel("Channel Number")
plt.ylabel("Counts")
plt.legend()
plt.show()
plt.show(block=True)

residuals = y_fit - gaussian_bg(x_fit, *params)
norm_residuals = residuals / np.sqrt(y_fit)
residual_errors = np.ones_like(norm_residuals)

plt.figure()
plt.plot(x_fit, norm_residuals, 'o')
plt.errorbar(
    x_fit, norm_residuals,
    yerr=residual_errors,  # ±1 for each residual
    fmt='o',              # marker
    color='red',       # marker color
    ecolor='gray',        # error bar color
    capsize=2,
    label='Residuals'
)
plt.axhline(0, linestyle='--')
plt.xlabel("Channel Number")
plt.ylabel("Normalized Residuals")
plt.show(block=True)


# Now can define an alpha window based on the acquired sigma