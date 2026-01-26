import mcphysics.data as m
import numpy as np

from pathlib import Path


def am_calibration():
    folder = Path("Calibration Files/Am_Calibration_Runs")

    for file in folder.iterdir():
        print(file.name)      # filename with extension (Ex: Am_Calibration000.Chn)
        data = m.load_chn("Calibration Files/Am_Calibration_Runs/"+file.name)

