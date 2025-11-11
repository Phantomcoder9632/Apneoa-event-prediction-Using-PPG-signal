import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import entropy
import pickle
import zipfile
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================
# Load Dataset
# =====================================
DATA_FILE = 'data\CNN_input_data.pickle'
ZIP_FILE = 'CNN_input_data.zip'

print("Loading dataset...")

if os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall('.')
    extracted_files = zip_ref.namelist()
    pickle_file = [f for f in extracted_files if f.endswith('.pickle') or f.endswith('.pkl')][0]
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
elif os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        dataset = pickle.load(f)

print(f"✅ Dataset loaded!")

# =====================================
# Parameters
# =====================================
FS = 100
WINDOW_SIZE_SEC = 30
WINDOW_SIZE = WINDOW_SIZE_SEC * FS
LEAD_TIME_SEC = 5
LEAD_TIME_SAMPLES = LEAD_TIME_SEC * FS
PREDICTION_WINDOW_SEC = 10
PREDICTION_WINDOW_SAMPLES = PREDICTION_WINDOW_SEC * FS

# =====================================
# Signal Processing
# =====================================
def extract_signal(patient_data):
    """Extract and flatten PPG signal"""
    signal_raw = np.array(patient_data['signal_pleth'].tolist())

    if signal_raw.ndim > 1:
        signal = np.nanmean(signal_raw, axis=1)
    else:
        signal = signal_raw

    # Remove NaN
    nan_mask = np.isnan(signal)
    if np.any(nan_mask):
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 0:
            signal[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, signal[valid_idx])

    return signal

def filter_ppg_light(signal, fs=100):
    """Light bandpass filter"""
    if len(signal) < 100:
        return signal

    nyquist = fs / 2
    b, a = scipy_signal.butter(2, [0.5/nyquist, 8.0/nyquist], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    return filtered

def calculate_segment_quality(segment, fs=100):
    """
    Calculate quality score for a segment
    Good baseline segments should have:
    - Low variability (stable)
    - Regular heart rate (60-90 bpm)
    - Low entropy (predictable)
    """
    if len(segment) < 100:
        return 0

    # Calculate heart rate from peaks
    distance = int(0.5 * fs)
    peaks, _ = scipy_signal.find_peaks(segment, distance=distance, prominence=0.1*np.std(segment))

    if len(peaks) < 3:
        return 0

    # Heart rate
    intervals = np.diff(peaks) / fs
    mean_hr = 60 / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0

    # Quality metrics
    std_normalized = np.std(segment) / (np.abs(np.mean(segment)) + 1e-6)
    range_normalized = (np.max(segment) - np.min(segment)) / (np.abs(np.mean(segment)) + 1e-6)

    # Good baseline: HR 60-90, low variability, regular
    hr_score = 1.0 if 60 <= mean_hr <= 90 else 0.5 if 50 <= mean_hr <= 100 else 0.1
    stability_score = 1.0 / (1.0 + std_normalized)  # Lower std is better
    regularity_score = 1.0 / (1.0 + range_normalized)  # Lower range is better

    quality = hr_score * stability_score * regularity_score

    return quality

def find_peaks_robust(signal, fs=100):
    """Find peaks in PPG signal"""
    distance = int(0.4 * fs)
    peaks, _ = scipy_signal.find_peaks(signal, distance=distance, prominence=0.1*np.std(signal))
    return peaks

def calculate_hrv_features(peaks, fs=100):
    """Calculate heart rate variability features"""
    if len(peaks) < 2:
        return {}

    intervals = np.diff(peaks) / fs

    if len(intervals) == 0:
        return {}

    features = {
        'mean_hr': 60 / np.mean(intervals) if np.mean(intervals) > 0 else 0,
        'std_rr': np.std(intervals),
        'rmssd': np.sqrt(np.mean(np.diff(intervals)**2)),
        'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
    }

    return features

def calculate_spectral_features(signal, fs=100):
    """Calculate frequency domain features"""
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)

    lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
    hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0

    features = {
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0
    }

    return features, freqs, psd

def calculate_statistical_features(signal):
    """Calculate statistical features"""
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'range': np.max(signal) - np.min(signal),
        'entropy': entropy(np.histogram(signal, bins=50)[0] + 1e-10)
    }
    return features

# =====================================
# IMPROVED Example Collection
# =====================================
print("\n" + "="*60)
print("Collecting examples with QUALITY-BASED selection")
print("="*60)

positive_examples = []
negative_examples = []
negative_qualities = []

for pat in dataset['patient'].unique():
    if len(positive_examples) >= 2 and len(negative_examples) >= 2:
        break

    temp = dataset[dataset['patient'] == pat].reset_index(drop=True)
    signal_raw = extract_signal(temp)
    signal = filter_ppg_light(signal_raw)

    labels = np.array(temp['anomaly'].tolist())
    if labels.ndim > 1:
        labels = labels.flatten()

    print(f"\n  Patient {pat}:")

    # POSITIVE: Pre-apnea windows
    if len(positive_examples) < 2:
        apnea_starts = np.where(np.diff(np.concatenate(([0], labels.astype(int)))) == 1)[0]

        for apnea_start in apnea_starts:
            if len(positive_examples) >= 2:
                break

            window_end = apnea_start - LEAD_TIME_SAMPLES
            window_start = window_end - WINDOW_SIZE

            if window_start >= 0 and window_end < len(signal):
                segment = signal[window_start:window_end]
                if np.std(segment) > 0.1:
                    positive_examples.append(segment)
                    print(f"    ✓ Positive example: std={np.std(segment):.2f}")

    # NEGATIVE: Find STABLE baseline periods (key improvement!)
    if len(negative_examples) < 2:
        # Scan for stable windows away from apnea
        stride = WINDOW_SIZE // 4  # 25% overlap
        candidates = []

        for i in range(0, len(signal) - WINDOW_SIZE - LEAD_TIME_SAMPLES - PREDICTION_WINDOW_SAMPLES, stride):
            window_end = i + WINDOW_SIZE
            pred_start = window_end + LEAD_TIME_SAMPLES
            pred_end = pred_start + PREDICTION_WINDOW_SAMPLES

            # Check no apnea in window or prediction period
            if (pred_end < len(labels) and
                not np.any(labels[i:window_end] == 1) and
                not np.any(labels[pred_start:pred_end] == 1)):

                segment = signal[i:window_end]

                if np.std(segment) > 0.1:
                    # Calculate quality score
                    quality = calculate_segment_quality(segment, FS)
                    candidates.append((segment, quality, i))

        # Sort by quality and take the best ones
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"    Found {len(candidates)} candidate windows")

        for segment, quality, idx in candidates[:2]:
            if len(negative_examples) >= 2:
                break

            # Calculate HR to report
            peaks = find_peaks_robust(segment, FS)
            if len(peaks) > 2:
                intervals = np.diff(peaks) / FS
                hr = 60 / np.mean(intervals) if len(intervals) > 0 else 0
                print(f"    ✓ Negative (quality={quality:.3f}, HR={hr:.1f}, std={np.std(segment):.2f})")
                negative_examples.append(segment)
                negative_qualities.append(quality)

print(f"\n{'='*60}")
print(f"Collected: {len(positive_examples)} positive, {len(negative_examples)} negative")
print(f"Negative qualities: {negative_qualities}")
print(f"{'='*60}")

if len(positive_examples) == 0 or len(negative_examples) == 0:
    raise ValueError("Not enough valid examples found!")

# Use best examples
pos_signal = positive_examples[0]
neg_signal = negative_examples[0]

# =====================================
# VISUALIZATION
# =====================================
print("\nCreating feature visualization...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.32)
fig.suptitle('CNN Feature Learning: Pre-Apnea vs Baseline (Quality-Selected Normal)\n(Real Data with Improved Selection)',
             fontsize=18, fontweight='bold', y=0.995)

time_axis = np.arange(len(pos_signal)) / FS

# Calculate features
pos_peaks = find_peaks_robust(pos_signal, FS)
neg_peaks = find_peaks_robust(neg_signal, FS)
pos_hrv = calculate_hrv_features(pos_peaks, FS)
neg_hrv = calculate_hrv_features(neg_peaks, FS)
pos_spec, pos_freqs, pos_psd = calculate_spectral_features(pos_signal, FS)
neg_spec, neg_freqs, neg_psd = calculate_spectral_features(neg_signal, FS)
pos_stats = calculate_statistical_features(pos_signal)
neg_stats = calculate_statistical_features(neg_signal)

# ROW 1: Raw signals
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_axis, pos_signal, 'r-', linewidth=1.3, alpha=0.85, label='Pre-Apnea')
ax1.plot(time_axis, neg_signal, 'g-', linewidth=1.3, alpha=0.85, label='Stable Baseline')
ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('PPG Amplitude', fontsize=12, fontweight='bold')
ax1.set_title('Raw PPG Comparison (Quality-Selected Baseline)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

# ROW 2: Peak detection
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_axis, pos_signal, 'r-', linewidth=1, alpha=0.7)
ax2.plot(time_axis[pos_peaks], pos_signal[pos_peaks], 'ro', markersize=7)
ax2.set_title(f'Pre-Apnea: {len(pos_peaks)} peaks\nHR: {pos_hrv.get("mean_hr", 0):.1f} bpm',
              fontsize=11, fontweight='bold', color='darkred')
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Amplitude', fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_axis, neg_signal, 'g-', linewidth=1, alpha=0.7)
ax3.plot(time_axis[neg_peaks], neg_signal[neg_peaks], 'go', markersize=7)
ax3.set_title(f'Baseline: {len(neg_peaks)} peaks\nHR: {neg_hrv.get("mean_hr", 0):.1f} bpm',
              fontsize=11, fontweight='bold', color='darkgreen')
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Amplitude', fontsize=10)
ax3.grid(True, alpha=0.3)

# HRV comparison
ax4 = fig.add_subplot(gs[1, 2])
hrv_metrics = ['mean_hr', 'std_rr', 'rmssd', 'cv']
hrv_labels = ['HR (bpm)', 'RR Std', 'RMSSD', 'CV']
pos_hrv_vals = [pos_hrv.get(m, 0) for m in hrv_metrics]
neg_hrv_vals = [neg_hrv.get(m, 0) for m in hrv_metrics]

x = np.arange(len(hrv_metrics))
width = 0.35
ax4.bar(x - width/2, pos_hrv_vals, width, label='Pre-Apnea', color='red', alpha=0.7)
ax4.bar(x + width/2, neg_hrv_vals, width, label='Baseline', color='green', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(hrv_labels, fontsize=9, rotation=15)
ax4.set_ylabel('Value', fontsize=10, fontweight='bold')
ax4.set_title('Heart Rate Variability', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# ROW 3: Frequency analysis
ax5 = fig.add_subplot(gs[2, 0])
ax5.semilogy(pos_freqs, pos_psd, 'r-', linewidth=2, alpha=0.8, label='Pre-Apnea')
ax5.semilogy(neg_freqs, neg_psd, 'g-', linewidth=2, alpha=0.8, label='Baseline')
ax5.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
ax5.set_ylabel('PSD', fontsize=10, fontweight='bold')
ax5.set_title('Power Spectral Density', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 2])

ax6 = fig.add_subplot(gs[2, 1])
bands = ['LF Power', 'HF Power', 'LF/HF']
pos_band_vals = [pos_spec['lf_power'], pos_spec['hf_power'], pos_spec['lf_hf_ratio']]
neg_band_vals = [neg_spec['lf_power'], neg_spec['hf_power'], neg_spec['lf_hf_ratio']]

x = np.arange(len(bands))
ax6.bar(x - width/2, pos_band_vals, width, label='Pre-Apnea', color='red', alpha=0.7)
ax6.bar(x + width/2, neg_band_vals, width, label='Baseline', color='green', alpha=0.7)
ax6.set_xticks(x)
ax6.set_xticklabels(bands, fontsize=9)
ax6.set_ylabel('Power', fontsize=10, fontweight='bold')
ax6.set_title('Frequency Bands', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

ax7 = fig.add_subplot(gs[2, 2])
f, t, Sxx = scipy_signal.spectrogram(pos_signal, FS, nperseg=128)
im = ax7.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='hot')
ax7.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
ax7.set_xlabel('Time (s)', fontsize=10)
ax7.set_title('Pre-Apnea Spectrogram', fontsize=11, fontweight='bold', color='darkred')
ax7.set_ylim([0, 5])
plt.colorbar(im, ax=ax7, label='dB')

# ROW 4: Statistical features
ax8 = fig.add_subplot(gs[3, 0])
stat_metrics = ['std', 'range', 'entropy']
stat_labels = ['Std Dev', 'Range', 'Entropy']
pos_stat_vals = [pos_stats['std'], pos_stats['range'], pos_stats['entropy']]
neg_stat_vals = [neg_stats['std'], neg_stats['range'], neg_stats['entropy']]

x = np.arange(len(stat_metrics))
ax8.bar(x - width/2, pos_stat_vals, width, label='Pre-Apnea', color='red', alpha=0.7)
ax8.bar(x + width/2, neg_stat_vals, width, label='Baseline', color='green', alpha=0.7)
ax8.set_xticks(x)
ax8.set_xticklabels(stat_labels, fontsize=9)
ax8.set_ylabel('Value', fontsize=10, fontweight='bold')
ax8.set_title('Statistical Features', fontsize=11, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3, axis='y')

# Difference signal
ax9 = fig.add_subplot(gs[3, 1])
min_len = min(len(pos_signal), len(neg_signal))
diff_signal = pos_signal[:min_len] - neg_signal[:min_len]
time_diff = np.arange(len(diff_signal)) / FS
ax9.plot(time_diff, diff_signal, 'purple', linewidth=1.3, alpha=0.8)
ax9.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax9.fill_between(time_diff, 0, diff_signal, alpha=0.3, color='purple')
ax9.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
ax9.set_ylabel('Difference', fontsize=10, fontweight='bold')
ax9.set_title('Difference Signal', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)

# Summary table
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')

hr_diff = abs(pos_hrv.get('mean_hr', 0) - neg_hrv.get('mean_hr', 0))
hrv_diff = abs(pos_hrv.get('std_rr', 0) - neg_hrv.get('std_rr', 0))
lfhf_diff = abs(pos_spec['lf_hf_ratio'] - neg_spec['lf_hf_ratio'])
std_diff = abs(pos_stats['std'] - neg_stats['std'])

summary_data = [
    ['Metric', 'Pre-Apnea', 'Baseline', 'Δ'],
    ['HR (bpm)', f"{pos_hrv.get('mean_hr', 0):.1f}", f"{neg_hrv.get('mean_hr', 0):.1f}", f"{hr_diff:.1f}"],
    ['HRV std', f"{pos_hrv.get('std_rr', 0):.3f}", f"{neg_hrv.get('std_rr', 0):.3f}", f"{hrv_diff:.3f}"],
    ['LF/HF', f"{pos_spec['lf_hf_ratio']:.2f}", f"{neg_spec['lf_hf_ratio']:.2f}", f"{lfhf_diff:.2f}"],
    ['Std Dev', f"{pos_stats['std']:.2f}", f"{neg_stats['std']:.2f}", f"{std_diff:.2f}"]
]

table = ax10.table(cellText=summary_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_data)):
    for j in range(4):
        if j == 0:
            table[(i, j)].set_facecolor('#e3f2fd')

ax10.set_title('Key Differences', fontsize=11, fontweight='bold', pad=20)

plt.savefig('ppg_features_fixed.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: ppg_features_fixed.png")
plt.show()

print("\n" + "="*60)
print("✅ Analysis Complete with QUALITY-BASED Selection!")
print("="*60)
print(f"\nFindings:")
print(f"  Pre-Apnea HR: {pos_hrv.get('mean_hr', 0):.1f} bpm")
print(f"  Baseline HR: {neg_hrv.get('mean_hr', 0):.1f} bpm")
print(f"  Pre-Apnea HRV: {pos_hrv.get('std_rr', 0):.3f}s")
print(f"  Baseline HRV: {neg_hrv.get('std_rr', 0):.3f}s")
print(f"  HR Change: {abs(pos_hrv.get('mean_hr', 0) - neg_hrv.get('mean_hr', 0)):.1f} bpm")
print("="*60)