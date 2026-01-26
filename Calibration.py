import mcphysics.data as m
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

def gaussian_bg(x, A, x0, sigma, B):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B

def am_calibration():
    folder = Path("Calibration Files/Am_Calibration_Runs")
    sigma_list = []
    for file in folder.iterdir():
        data = m.load_chn(file)

        # Find peak
        max_index = data['Counts'].argmax()
        max_count = data['Counts'][max_index]

        # Fit window
        fit_min = max_index - 8
        fit_max = max_index + 8

        mask = (data['Channel'] >= fit_min) & (data['Channel'] <= fit_max)

        # Initial guesses
        x_fit = data['Channel'][mask]
        y_fit = data['Counts'][mask]
        B0 = np.median(y_fit)
        sigma0 = 7
        A0 = max_count - B0
        p0 = [A0, max_index, sigma0, B0]
        try:
            params, cov = curve_fit(gaussian_bg, x_fit, y_fit, p0=p0)
        except RuntimeError:
            print(f"Fit failed for {file.name}")
            continue
        A, x0, sigma, B = params
        sigma_list.append(sigma)

am_calibration()


