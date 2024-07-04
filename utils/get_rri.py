import neurokit2 as nk
import numpy as np
def get_rri(ecg_signal: np.ndarray, sampling_rate: int|None = 2000) -> np.ndarray:
    ecg, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    r_peaks = info["ECG_R_Peaks"]

    rri = np.diff(r_peaks) / sampling_rate * 1000 

    return rri