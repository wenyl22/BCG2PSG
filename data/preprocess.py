import os
import numpy as np
import neurokit2 as nk
from scipy.stats import zscore
import matplotlib.pyplot as plt
def visualize(bcg, ecg = None, rsp = None):
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1) 
    plt.plot(bcg, label='BCG')
    plt.legend()
    plt.title('BCG Data')

    plt.subplot(3, 1, 2)  
    plt.plot(ecg, label='ECG')
    plt.legend()
    plt.title('ECG Data')

    plt.subplot(3, 1, 3)  
    plt.plot(rsp, label='RSP')
    plt.legend()
    plt.title('RSP Data')

    plt.tight_layout()
    plt.savefig("./data.png")

def Preprocess(data_root, src, path, qualities):
    if os.path.exists(f"{data_root}/{src}/psg_quality_ecg_ECG II.npz") == False:
        return
    if os.path.exists(f"{data_root}/{src}/psg_feature_abd_Effort ABD_respiration.npz") == False:
        return
    with np.load(f"{data_root}/{src}/bcg_feature_raw.npz") as f:
        bcg_feature = f["raw"]
        bcg_freq = f["fs"]
    with np.load(f"{data_root}/{src}/psg_feature_ecg_ECG II_raw.npz") as f:
        ecg_feature = f["ecg"]
        ecg_freq = f["fs"]
    with np.load(f"{data_root}/{src}/bcg_quality_bio_freq.npz") as f:
        bcg_quality = f["quality"]
        bcg_quality_freq = f["fs"]
    with np.load(f"{data_root}/{src}/psg_quality_ecg_ECG II.npz") as f:
        ecg_quality = f["quality"]
        ecg_quality_freq = f["fs"]
    with np.load(f"{data_root}/{src}/psg_feature_abd_Effort ABD_respiration.npz") as f:
        rsp_feature = f["respiration"]
        rsp_freq = f["fs"]
    with np.load(f"{data_root}/{src}/psg_quality_abd_Effort ABD.npz") as f:
        rsp_quality = f["quality"]
        rsp_quality_freq = f["fs"]
    print(bcg_freq, ecg_freq, rsp_freq)
    for l in range(0, int(len(bcg_quality)//bcg_quality_freq) - 50, 15):
        r = l + 30
        mean_bcg_quality = np.mean(bcg_quality[int(l * bcg_quality_freq) : int(r * bcg_quality_freq)]) 
        mean_ecg_quality = np.mean(ecg_quality[int(l * ecg_quality_freq) : int(r * ecg_quality_freq)])
        mean_rsp_quality = np.mean(rsp_quality[int(l * rsp_quality_freq) : int(r * rsp_quality_freq)])
        if mean_bcg_quality < 0.6 or mean_rsp_quality < 0.5 or mean_ecg_quality < 0.85:
            continue
        # qualities[0].append(mean_bcg_quality)
        # qualities[1].append(mean_ecg_quality)
        # qualities[2].append(mean_rsp_quality)
        rsp = rsp_feature[l * rsp_freq : r * rsp_freq].astype(np.float32)
        rsp = nk.signal_resample(rsp, desired_length = 3000)
        rsp = nk.signal_filter(rsp, sampling_rate=100, lowcut=0.05, highcut = 3).astype(np.float32)
        rsp = zscore(rsp)

        bcg = bcg_feature[l * bcg_freq : r * bcg_freq].astype(np.float32)
        bcg = nk.signal_resample(bcg, desired_length = 3000)
        bcg = nk.signal_filter(bcg, sampling_rate=100, lowcut=0.1, highcut=25).astype(np.float32)
        bcg = zscore(bcg)

        ecg = ecg_feature[l * ecg_freq : r * ecg_freq].astype(np.float32)
        ecg = nk.signal_resample(ecg, desired_length = 3000)
        #ecg = nk.signal_filter(ecg, sampling_rate=100, lowcut=0.1, highcut=25).astype(np.float32)
        ecg = nk.ecg_clean(ecg, sampling_rate = 100, method = "biosppy")
        ecg, _ = nk.ecg_invert(ecg, sampling_rate = 100)
        ecg = zscore(ecg)
        # visualize(bcg, ecg, rsp)
        np.save(f"{data_root}/{src}/{l}_rsp.npy", rsp)
        np.save(f"{data_root}/{src}/{l}_bcg.npy", bcg)
        np.save(f"{data_root}/{src}/{l}_ecg.npy", ecg)
        path.append(f"{data_root}/{src}/{l}")
