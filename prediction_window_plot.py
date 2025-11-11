import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import pickle
import os
import sys
import tensorflow as tf

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# =====================================
# Configuration
# =====================================
DATA_FILE = 'data/CNN_input_data.pickle' 
MODEL_PATH = 'my_final_apnea_predictor.keras'
TEST_PATIENT_IDS = ['7', '17', '18', '23', '25', '30'] 
OUTPUT_DIR = 'model_true_positive_plots' # New directory for these plots
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Prediction window parameters
OVERVIEW_WINDOW_SEC = 30
LEAD_TIME_SEC = 5
PREDICTION_WINDOW_SEC = 10
NUM_EXAMPLES_PER_PATIENT = 6 # 6 plots per patient (3x2 grid)

# =====================================
# Signal Processing Functions (Corrected for OSASUD format)
# =====================================

def extract_full_patient_signals(patient_data: pd.DataFrame):
    ppg_signal_list = []
    current_fs = 100 # Default fallback
    valid_ppg_entries = [e for e in patient_data['signal_pleth'].tolist() 
                         if isinstance(e, (np.ndarray, list)) and len(e) > 0]
    if not valid_ppg_entries:
        raise ValueError("No valid PPG data found in 'signal_pleth'.")
    current_fs = len(valid_ppg_entries[0])
    for entry in patient_data['signal_pleth'].tolist():
        if isinstance(entry, (np.ndarray, list)) and len(entry) == current_fs:
            ppg_signal_list.append(np.array(entry, dtype=float))
        else:
            ppg_signal_list.append(np.full(current_fs, np.nan))
    ppg_signal_full = np.concatenate(ppg_signal_list)
    anomaly_labels_per_second = patient_data['anomaly'].values.astype(int)
    anomaly_labels_upsampled = np.repeat(anomaly_labels_per_second, current_fs)
    nan_mask = np.isnan(ppg_signal_full)
    if np.any(nan_mask):
        nan_indices = np.where(nan_mask)[0]
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 2:
            ppg_signal_full[nan_indices] = np.interp(nan_indices, valid_indices, ppg_signal_full[valid_indices])
        else:
            ppg_signal_full[nan_mask] = 0 
    return ppg_signal_full, anomaly_labels_upsampled, current_fs

def filter_ppg_signal(signal: np.ndarray, fs: int, lowcut=0.5, highcut=8.0):
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0: high = 0.99 
    try:
        b, a = scipy_signal.butter(4, [low, high], btype='band')
    except ValueError as e:
        print(f"--- ERROR: Could not design filter. fs={fs}, low={low}, high={high}. Error: {e} ---")
        return signal
    return scipy_signal.filtfilt(b, a, signal)

def is_valid_segment(signal: np.ndarray, start_idx: int, end_idx: int, min_std=0.5):
    if start_idx < 0 or end_idx > len(signal): return False
    segment = signal[start_idx:end_idx]
    if len(segment) == 0: return False
    if not np.all(np.isfinite(segment)): return False
    return np.std(segment) > min_std

# =====================================
# Load Dataset and Get FS Map
# =====================================
print("Loading dataset...")
try:
    dataset = pd.read_pickle(DATA_FILE)
    print("✅ Dataset loaded!")
except Exception as e:
    print(f"--- ERROR loading {DATA_FILE}: {e} ---")
    sys.exit()

patient_fs_map = {}
print("Finding sampling rates for test patients...")
all_patients = dataset['patient'].unique()
for pat_id in all_patients:
    temp = dataset[dataset['patient'] == str(pat_id)]
    if temp.empty: temp = dataset[dataset['patient'] == int(pat_id)]
    if temp.empty: continue
    try:
        valid_ppg = [e for e in temp['signal_pleth'].tolist() if isinstance(e, (np.ndarray, list)) and len(e) > 0]
        if not valid_ppg: continue
        FS_pat = len(valid_ppg[0])
        patient_fs_map[str(pat_id)] = FS_pat
    except Exception as e:
        print(f"Error getting FS for Patient {pat_id}: {e}")

try:
    UNIVERSAL_FS = patient_fs_map[TEST_PATIENT_IDS[0]]
    print(f"✅ Using universal Sampling Rate (FS): {100} Hz")
except KeyError:
    print(f"--- ERROR: Could not determine FS for test patients. Exiting. ---")
    sys.exit()

# =====================================
# Load Trained Model
# =====================================
print(f"\nLoading trained model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"--- ERROR loading model: {e} ---"); sys.exit()

# =====================================
# MAIN PLOTTING LOOP
# =====================================
print("\n" + "="*60)
print("Finding and Plotting TRUE POSITIVE Predictions for Test Patients")
print("="*60)

for pat_id in TEST_PATIENT_IDS:
    pat_id = str(pat_id)
    if pat_id not in patient_fs_map:
        print(f"\nSkipping Patient {pat_id}: Data not found.")
        continue
    
    FS_pat = patient_fs_map[pat_id]
    if FS_pat != UNIVERSAL_FS:
        print(f"--- WARNING: Skipping Patient {pat_id}: FS ({FS_pat}Hz) does not match model FS ({UNIVERSAL_FS}Hz). ---")
        continue

    OVERVIEW_SAMPLES = OVERVIEW_WINDOW_SEC * FS_pat
    LEAD_SAMPLES = LEAD_TIME_SEC * FS_pat
    PREDICTION_SAMPLES = PREDICTION_WINDOW_SEC * FS_pat

    print(f"\n--- Processing Patient {pat_id} ---")
    temp_df = dataset[dataset['patient'] == pat_id]
    if temp_df.empty: temp_df = dataset[dataset['patient'] == int(pat_id)]
        
    try:
        signal, labels_upsampled, _ = extract_full_patient_signals(temp_df)
        signal = filter_ppg_signal(signal, fs=FS_pat)
    except Exception as e:
        print(f"  ERROR processing signals for {pat_id}: {e}. Skipping plot.")
        continue

    # --- 1. Find all valid pre-apnea windows (Ground Truth) ---
    valid_pre_apnea_start_samples = [] # Stores start sample of OVERVIEW window
    for i in range(0, len(signal) - (OVERVIEW_SAMPLES + LEAD_SAMPLES + PREDICTION_SAMPLES), FS_pat): 
        overview_start_abs = i
        overview_end_abs = overview_start_abs + OVERVIEW_SAMPLES
        apnea_check_start_abs = overview_end_abs + LEAD_SAMPLES
        apnea_check_end_abs = apnea_check_start_abs + PREDICTION_SAMPLES
        
        if (not np.any(labels_upsampled[overview_start_abs:overview_end_abs] == 1) and
            np.any(labels_upsampled[apnea_check_start_abs:apnea_check_end_abs] == 1) and
            is_valid_segment(signal, overview_start_abs, overview_end_abs)):
            valid_pre_apnea_start_samples.append(overview_start_abs)
            
    if not valid_pre_apnea_start_samples:
        print(f"  Found 0 total valid pre-apnea windows for this patient.")
        continue
    
    # --- 2. Filter for TRUE POSITIVES (Model Prediction) ---
    true_positive_windows = [] # Store (start_sample, probability)
    
    print(f"  Found {len(valid_pre_apnea_start_samples)} total pre-apnea windows. Running model on them...")
    
    # Create a batch of all valid overview segments for this patient
    batch_X = []
    for overview_start_abs in valid_pre_apnea_start_samples:
        overview_segment = signal[overview_start_abs : overview_start_abs + OVERVIEW_SAMPLES]
        segment_norm = (overview_segment - np.mean(overview_segment)) / (np.std(overview_segment) + 1e-6)
        batch_X.append(segment_norm)
    
    # Reshape for model: (Batch, Samples, Channels)
    batch_X_np = np.array(batch_X).reshape(len(batch_X), OVERVIEW_SAMPLES, 1)
    
    # Get all predictions in one go (much faster)
    try:
        with tf.device('/cpu:0'): # Ensure prediction runs
            all_probs = model.predict(batch_X_np, verbose=0).flatten() # Get all probabilities
    except Exception as e:
        print(f"  ERROR during batch prediction: {e}. Skipping patient.")
        continue

    # Filter for predictions >= 0.5
    for i, overview_start_abs in enumerate(valid_pre_apnea_start_samples):
        prob = all_probs[i]
        if prob >= 0.5:
            true_positive_windows.append((overview_start_abs, prob)) # Store start sample and its probability

    print(f"  Model CORRECTLY PREDICTED {len(true_positive_windows)} of them (True Positives).")

    if not true_positive_windows:
        print("  No True Positive predictions found to plot for this patient.")
        continue

    # --- 3. Plot the True Positive Examples ---
    num_to_plot = min(len(true_positive_windows), NUM_EXAMPLES_PER_PATIENT)
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), squeeze=False)
    fig.suptitle(f'Patient {pat_id}: CORRECT Pre-Apnea Predictions (True Positives: {len(true_positive_windows)})', 
                 fontsize=18, fontweight='bold', y=1.03)
    axes_flat = axes.flatten()

    np.random.shuffle(true_positive_windows) # Shuffle the TP examples
    plot_examples = true_positive_windows[:num_to_plot]

    for i, (overview_start_abs, pred_prob) in enumerate(plot_examples):
        ax = axes_flat[i]
        
        overview_end_abs = overview_start_abs + OVERVIEW_SAMPLES
        pred_start_abs = overview_end_abs + LEAD_SAMPLES
        pred_end_abs = pred_start_abs + PREDICTION_SAMPLES
        
        # Calculate time boundaries in seconds (Absolute Time)
        overview_start_time = overview_start_abs / FS_pat
        overview_end_time = overview_end_abs / FS_pat
        pred_start_time = pred_start_abs / FS_pat
        pred_end_time = pred_end_abs / FS_pat
        
        actual_apnea_onset_sample = None
        apnea_in_pred_window = np.where(labels_upsampled[pred_start_abs:pred_end_abs] == 1)[0]
        if len(apnea_in_pred_window) > 0:
            actual_apnea_onset_sample = pred_start_abs + apnea_in_pred_window[0]

        context_samples = 5 * FS_pat 
        viz_start_abs = max(0, overview_start_abs - context_samples)
        viz_end_abs = min(len(signal), pred_end_abs + context_samples)
        
        segment_viz = signal[viz_start_abs:viz_end_abs]
        
        # --- MODIFIED: Use Absolute Time for X-axis ---
        time_abs = (np.arange(len(segment_viz)) + viz_start_abs) / FS_pat 
        
        ax.plot(time_abs, segment_viz, 'b-', linewidth=0.9, label='Filtered PPG', zorder=2)
        
        # --- Shade regions (using Absolute Time boundaries) ---
        ax.axvspan(overview_start_time, overview_end_time, 
                   alpha=0.15, color='green', label='Overview (30s Input)')
        ax.axvspan(overview_end_time, pred_start_time, 
                   alpha=0.2, color='orange', label='Lead Time (5s)')
        ax.axvspan(pred_start_time, pred_end_time, 
                   alpha=0.3, color='lightcoral', label='Prediction Window (10s Target)')
        
        labels_viz = labels_upsampled[viz_start_abs:viz_end_abs] == 1
        ax.fill_between(time_abs, segment_viz.min(), segment_viz.max(), where=labels_viz,
                         color='red', alpha=0.4, label='Actual Apnea Event', zorder=3)
        
        if actual_apnea_onset_sample is not None:
            # Apnea onset is now plotted using its absolute time
            apnea_onset_abs_time = actual_apnea_onset_sample / FS_pat
            ax.axvline(apnea_onset_abs_time, color='darkred', linestyle='--', linewidth=2, 
                       label=f'Apnea Onset (t={apnea_onset_abs_time:.1f}s)', zorder=4)

        # --- Display the CORRECT Model Prediction ---
        pred_text = f'Model Pred: APNEA' # We know this is a True Positive
        pred_color = 'darkred'
        
        ax.text(0.02, 0.95, pred_text + f' (Prob={pred_prob:.2f})', transform=ax.transAxes,
                 fontsize=11, fontweight='bold', color=pred_color, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=pred_color, lw=1))

        # --- Plot Aesthetics ---
        ax.set_title(f'Correct Prediction Example {i+1} (Patient {pat_id})', 
                     fontsize=13, fontweight='bold', pad=10)
        # --- MODIFIED X-AXIS LABEL ---
        ax.set_xlabel('Absolute Time Since Recording Start (seconds)', fontsize=11) 
        ax.set_ylabel('PPG Amplitude', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        y_min, y_max = np.nanmin(segment_viz), np.nanmax(segment_viz)
        y_range = y_max - y_min
        if y_range > 1e-6:
             ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.2) 

    # Fill any empty subplots
    for j in range(num_to_plot, NUM_EXAMPLES_PER_PATIENT):
        axes_flat[j].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    filename = os.path.join(OUTPUT_DIR, f'patient_{pat_id}_TRUE_POSITIVES_ABS_TIME.png') # Added ABS_TIME to filename
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close(fig) # Close the figure

print("\n" + "="*60)
print("✅ All TRUE POSITIVE prediction plots generated (using Absolute Time)! ")
print("="*60)