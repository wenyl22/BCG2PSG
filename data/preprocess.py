import os
import numpy as np
import neurokit2 as nk
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pywt

def visualize_pywt(coeffs):
    plt.figure(figsize=(12, 24))
    for i in range(12):
        plt.subplot(12, 1, i+1)
        plt.plot(coeffs[i])
    plt.tight_layout()
    plt.savefig(f"./pywt/cur.png")
    plt.close()
    exit(0)

def visualize_wide_and_narrow(wide, narrow):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(wide)
    plt.title('Wide')
    plt.subplot(1, 2, 2)
    plt.plot(narrow)
    plt.title('Narrow')
    plt.tight_layout()
    plt.savefig(f"./wide_and_narrow.png")
    plt.close()
    exit(0)

def Preprocess(data_root, save_root, src, path, qualities):
    if os.path.exists(f"{data_root}/{src}/psg_quality_ecg_ECG II.npz") == False:
        return
    if os.path.exists(f"{data_root}/{src}/psg_feature_abd_Effort ABD_respiration.npz") == False:
        return
    with np.load(f"{data_root}/{src}/bcg_feature_raw.npz") as f:
        bcg_feature = f["raw"]
        bcg_freq = f["fs"]
    with np.load(f"{data_root}/{src}/psg_feature_ecg_ECG II_raw.npz") as f:
#        print(f.files)
        ecg_feature = f["raw"]
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
    with open(f"{data_root}/{src}/trigger_quality.txt") as f:
        trigger_quality = float(f.read().split(",")[1])
    
    for l in range(60, int(len(bcg_quality)//bcg_quality_freq) - 90, 30):
        r = l + 60
        mean_bcg_quality = np.mean(bcg_quality[int(l * bcg_quality_freq) : int(r * bcg_quality_freq)]) 
        mean_ecg_quality = np.mean(ecg_quality[int(l * ecg_quality_freq) : int(r * ecg_quality_freq)])
        mean_rsp_quality = np.mean(rsp_quality[int(l * rsp_quality_freq) : int(r * rsp_quality_freq)])
        # qualities[0].append(trigger_quality)
        # qualities[1].append(mean_bcg_quality)
        # qualities[2].append(mean_ecg_quality)
        # qualities[3].append(mean_rsp_quality)
        if mean_bcg_quality < 0.6 or mean_rsp_quality < 0.5 or mean_ecg_quality < 0.85 or trigger_quality < 0.5:
            continue
        path.append(f"/nvme3/wenyule/bcg2psg/{src}/{l}")

        # bcg = bcg_feature[l * bcg_freq : r * bcg_freq].astype(np.float32)
        # coeffs = pywt.wavedec(bcg, 'haar', level = 10)
        # temp = np.zeros((12, 6000))
        # for i in range(1, 12):
        #     temp[i] = zscore(nk.signal_resample(coeffs[i - 1], desired_length = 6000))
        # temp[0] = nk.signal_resample(bcg, desired_length = 6000)
        # temp[0] = nk.signal_filter(temp[0], sampling_rate=100, lowcut=0.1, highcut=25).astype(np.float32)
        # temp[0] = zscore(temp[0])
        # # visualize_pywt(temp)
        # # exit(0)

        # rsp = rsp_feature[l * rsp_freq : r * rsp_freq].astype(np.float32)
        # rsp = nk.signal_resample(rsp, desired_length = 6000)
        # rsp = nk.signal_filter(rsp, sampling_rate=100, lowcut=0.05, highcut = 3).astype(np.float32)
        # rsp = zscore(rsp)

        # ecg = ecg_feature[l * ecg_freq : r * ecg_freq].astype(np.float32)
        # ecg = nk.signal_resample(ecg, desired_length = 6000)
        # ecg = nk.ecg_clean(ecg, sampling_rate = 100, method = "biosppy").astype(np.float32)
        # ecg, _ = nk.ecg_invert(ecg, sampling_rate = 100)
        # ecg = zscore(ecg)
        # np.save(f"{save_root}/{src}/{l}_rsp.npy", rsp)
        # np.save(f"{save_root}/{src}/{l}_bcg.npy", temp)
        # np.save(f"{save_root}/{src}/{l}_ecg.npy", ecg)
