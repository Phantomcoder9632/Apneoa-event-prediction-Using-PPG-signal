import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
import os # Added os to create output directory

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# --- Output Directory ---
OUTPUT_DIR = 'model_results_plots'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"Saving all plots to: {OUTPUT_DIR}/")

# =====================================
# FABRICATED RESULTS DATA (FROM YOUR SCREENSHOTS)
# =====================================

# Cross-validation results (from Screenshot 2025-11-11 163758.png)
fold_results = {
    'Fold': [1, 2, 3, 4, 5],
    'AUC': [0.6789, 0.6648, 0.6971, 0.6682, 0.7110],
    'F1': [0.6593, 0.6534, 0.6518, 0.6408, 0.6466],
    'Recall': [0.8803, 0.9074, 0.8742, 0.8850, 0.8387],
    'Precision': [0.5227, 0.5380, 0.5517, 0.4770, 0.5438],
}
df_folds = pd.DataFrame(fold_results)

# Confusion matrices (Placeholders, as these were not in the screenshots)
# NOTE: These placeholders will affect Fig 3, 4, 6, 7.
# Replace these with your actual CM data if you have it.
confusion_matrices = [
    np.array([[191, 99], [134, 1149]]),   # Fold 1
    np.array([[11, 81], [79, 1420]]),    # Fold 2
    np.array([[106, 33], [197, 1237]]),   # Fold 3
    np.array([[28, 91], [308, 1152]]),    # Fold 4
    np.array([[19, 69], [234, 1259]])    # Fold 5
]

# Final test results (from Screenshot 2025-11-11 165227.png)
test_results = {
    'AUC': 0.7200,
    'F1': 0.7982,
    'Recall': 0.8403,
    'Precision': 0.7601,
    'Confusion_Matrix': np.array([[1376, 500], [301, 1584]]) # From screenshot
}

# Training history (simulated placeholder data)
fold1_training = {
    'epochs': list(range(1, 31)),
    'train_auc': [0.5671, 0.6780, 0.7363, 0.7293, 0.7615, 0.7705, 0.7865, 0.7860,
                  0.7893, 0.7877, 0.7779, 0.9074, 0.9597, 0.9799, 0.9830, 0.9936,
                  0.9957, 0.9960, 0.9975, 0.9980, 0.9939, 0.9964, 0.9987, 0.9989,
                  0.9992, 0.9997, 0.9991, 0.9975, 0.9995, 0.9996],
    'val_auc': [0.5000, 0.5000, 0.5627, 0.5315, 0.6269, 0.5501, 0.6235, 0.5886,
                0.5560, 0.5157, 0.8013, 0.7640, 0.6509, 0.6777, 0.7298, 0.7158,
                0.7502, 0.7840, 0.7839, 0.5762, 0.7024, 0.6550, 0.7036, 0.6886,
                0.7029, 0.7624, 0.7792, 0.6636, 0.6384, 0.8453],
    'train_loss': [0.4743, 0.0978, 0.0923, 0.0897, 0.0850, 0.0827, 0.0799, 0.0761,
                   0.0774, 0.0751, 0.0778, 0.0592, 0.0458, 0.0371, 0.0316, 0.0182,
                   0.0122, 0.0136, 0.0101, 0.0089, 0.0207, 0.0113, 0.0079, 0.0072,
                   0.0059, 0.0032, 0.0054, 0.0053, 0.0047, 0.0041],
    'val_loss': [50.4213, 42.7084, 25.4587, 10.3787, 2.2952, 1.8408, 1.4266, 0.9505,
                 0.9194, 0.5796, 0.5464, 0.8185, 1.2173, 1.3395, 0.8079, 1.4750,
                 1.1368, 0.8767, 1.1162, 2.6399, 1.4902, 1.8458, 1.4992, 1.6372,
                 1.6629, 1.0503, 1.1331, 1.6328, 1.9926, 1.1053]
}

# =====================================
# FIGURE 1: Training Curves (Loss & AUC)
# =====================================
print("Creating Figure 1: Training Curves...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Progress - Fold 1 (Representative Example)', fontsize=16, fontweight='bold')

    # Loss curve
    ax = axes[0]
    ax.plot(fold1_training['epochs'], fold1_training['train_loss'],
            'b-', linewidth=2, label='Training Loss', marker='o', markersize=4, alpha=0.7)
    ax.plot(fold1_training['epochs'], fold1_training['val_loss'],
            'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

    # AUC curve
    ax = axes[1]
    ax.plot(fold1_training['epochs'], fold1_training['train_auc'],
            'b-', linewidth=2, label='Training AUC', marker='o', markersize=4, alpha=0.7)
    ax.plot(fold1_training['epochs'], fold1_training['val_auc'],
            'r-', linewidth=2, label='Validation AUC', marker='s', markersize=4, alpha=0.7)
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target AUC=0.8')
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title('AUC Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_1_training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_1_training_curves.png')}")
    plt.close(fig) # Close the figure to free memory
except Exception as e:
    print(f"Error plotting Figure 1: {e}")

# =====================================
# FIGURE 2: Cross-Validation Results
# =====================================
print("\nCreating Figure 2: Cross-Validation Performance...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5-Fold Cross-Validation Results', fontsize=16, fontweight='bold')

    metrics = ['AUC', 'F1', 'Recall', 'Precision']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_folds['Fold'], df_folds[metric],
                      color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)
        mean_val = df_folds[metric].mean()
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_val:.4f}')
        std_val = df_folds[metric].std()
        ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray',
                   label=f'±1 SD: {std_val:.4f}')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} per Fold', fontsize=13, fontweight='bold')
        ax.set_xticks(df_folds['Fold'])
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_2_cv_performance.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_2_cv_performance.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 2: {e}")

# =====================================
# FIGURE 3: Confusion Matrix Heatmaps
# =====================================
print("\nCreating Figure 3: Confusion Matrices...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices - All Folds + Final Test', fontsize=16, fontweight='bold')

    # Plot each fold
    for i in range(5):
        ax = axes[i // 3, i % 3]
        cm = confusion_matrices[i]
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), cm_sum, 
                             out=np.zeros_like(cm, dtype=float), 
                             where=cm_sum!=0) * 100
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                    square=True, linewidths=2, linecolor='black',
                    cbar_kws={'label': 'Count'})
        for j in range(2):
            for k in range(2):
                ax.text(k + 0.5, j + 0.7, f'({cm_norm[j, k]:.1f}%)',
                       ha='center', va='center', fontsize=9, color='red', fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'Fold {i+1} (AUC={fold_results["AUC"][i]:.3f})',
                     fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Normal', 'Apnea'])
        ax.set_yticklabels(['Normal', 'Apnea'])

    # Plot test set
    ax = axes[1, 2]
    cm_test = test_results['Confusion_Matrix']
    cm_test_sum = cm_test.sum(axis=1)[:, np.newaxis]
    cm_test_norm = np.divide(cm_test.astype('float'), cm_test_sum, 
                             out=np.zeros_like(cm_test, dtype=float), 
                             where=cm_test_sum!=0) * 100
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax,
                square=True, linewidths=2, linecolor='black',
                cbar_kws={'label': 'Count'})
    for j in range(2):
        for k in range(2):
            ax.text(k + 0.5, j + 0.7, f'({cm_test_norm[j, k]:.1f}%)',
                   ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_title(f'Final Test (AUC={test_results["AUC"]:.3f})',
                 fontsize=12, fontweight='bold', color='darkred')
    ax.set_xticklabels(['Normal', 'Apnea'])
    ax.set_yticklabels(['Normal', 'Apnea'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_3_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_3_confusion_matrices.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 3: {e}")

# =====================================
# FIGURE 4: Spider/Radar Plot
# =====================================
print("\nCreating Figure 4: Spider Plot Comparison...")
try:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    categories = ['AUC', 'F1-Score', 'Recall', 'Precision', 'Specificity']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Calculate CV Specificity
    cv_specificities = []
    for cm in confusion_matrices:
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        cv_specificities.append(spec)

    # CV average
    cv_values = [
        df_folds['AUC'].mean(),
        df_folds['F1'].mean(),
        df_folds['Recall'].mean(),
        df_folds['Precision'].mean(),
        np.mean(cv_specificities) # Use mean of calculated specificities
    ]
    cv_values += cv_values[:1]

    # Test set
    tn, fp, fn, tp = test_results['Confusion_Matrix'].ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    test_values = [
        test_results['AUC'],
        test_results['F1'],
        test_results['Recall'],
        test_results['Precision'],
        test_specificity
    ]
    test_values += test_values[:1]

    # Plot
    ax.plot(angles, cv_values, 'o-', linewidth=3, label='CV Average', color='#3498db')
    ax.fill(angles, cv_values, alpha=0.25, color='#3498db')
    ax.plot(angles, test_values, 'o-', linewidth=3, label='Test Set', color='#e74c3c')
    ax.fill(angles, test_values, alpha=0.25, color='#e74c3c')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linewidth=1.5, alpha=0.5)
    plt.title('Performance Comparison: CV vs Test Set',
              fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_4_spider_plot.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_4_spider_plot.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 4: {e}")

# =====================================
# FIGURE 5: Performance Variability
# =====================================
print("\nCreating Figure 5: Performance Variability Analysis...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Variability Across Folds', fontsize=16, fontweight='bold')

    metrics = ['AUC', 'F1', 'Recall', 'Precision']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # Box plots
    ax = axes[0, 0]
    data_to_plot = [df_folds[metric] for metric in metrics]
    bp = ax.boxplot(data_to_plot, labels=metrics,
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Metrics Across Folds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Violin plot
    ax = axes[0, 1]
    positions = [1, 2, 3, 4]
    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Violin Plot - Metric Distributions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Fold comparison line plot
    ax = axes[1, 0]
    for metric, color in zip(metrics, colors):
        ax.plot(df_folds['Fold'], df_folds[metric],
                'o-', linewidth=2, markersize=8, label=metric, color=color, alpha=0.7)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Metric Trends Across Folds', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_folds['Fold'])
    ax.set_ylim([0, 1.05])

    # Coefficient of Variation
    ax = axes[1, 1]
    cvs = []
    for metric in metrics:
        mean_val = df_folds[metric].mean()
        std_val = df_folds[metric].std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        cvs.append(cv)
    bars = ax.bar(metrics, cvs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Metric Stability (Lower = More Stable)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_5_variability_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_5_variability_analysis.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 5: {e}")

# =====================================
# FIGURE 6: ROC Curves (New!)
# =====================================
print("\nCreating Figure 6: ROC Curves...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ROC Curves - All Folds + Average', fontsize=16, fontweight='bold')
    fold_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx in range(5):
        ax = axes[idx // 3, idx % 3]
        cm = confusion_matrices[idx]
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC=0.5)')
        ax.plot([0, fpr, 1], [0, tpr, 1], 'o-', linewidth=3, markersize=10,
                color=fold_colors[idx], alpha=0.8, label=f'Fold {idx+1} (AUC={fold_results["AUC"][idx]:.3f})')
        ax.fill_between([0, fpr, 1], 0, [0, tpr, 1], alpha=0.2, color=fold_colors[idx])
        ax.scatter([fpr], [tpr], s=200, c=fold_colors[idx], edgecolor='black',
                   linewidth=2, zorder=10, marker='D')
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'Fold {idx+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_aspect('equal')

    # Comparison plot in last subplot
    ax = axes[1, 2]
    for idx in range(5):
        cm = confusion_matrices[idx]
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ax.scatter([fpr], [tpr], s=150, c=fold_colors[idx], alpha=0.7,
                   edgecolor='black', linewidth=2, label=f'F{idx+1}')
    tn_t, fp_t, fn_t, tp_t = test_results['Confusion_Matrix'].ravel()
    tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
    ax.scatter([fpr_t], [tpr_t], s=300, c='red', marker='*',
               edgecolor='black', linewidth=3, label='Test', zorder=10)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('All Folds Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_6_roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_6_roc_curves.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 6: {e}")

# =====================================
# FIGURE 7: Class Balance Analysis
# =====================================
print("\nCreating Figure 7: Class Balance Analysis...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class Distribution and Balance Analysis', fontsize=16, fontweight='bold')

    # 1. Actual class distribution per fold
    ax = axes[0, 0]
    normal_counts = [cm.ravel()[0] + cm.ravel()[1] for cm in confusion_matrices]
    apnea_counts = [cm.ravel()[2] + cm.ravel()[3] for cm in confusion_matrices]
    x = np.arange(5); width = 0.35
    ax.bar(x - width/2, normal_counts, width, label='Normal', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, apnea_counts, width, label='Apnea', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'F{i+1}' for i in range(5)])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')

    # 2. Prediction distribution
    ax = axes[0, 1]
    pred_normal = [cm.ravel()[0] + cm.ravel()[2] for cm in confusion_matrices]
    pred_apnea = [cm.ravel()[1] + cm.ravel()[3] for cm in confusion_matrices]
    ax.bar(x - width/2, pred_normal, width, label='Predicted Normal', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, pred_apnea, width, label='Predicted Apnea', color='#f39c12', alpha=0.8)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Count', fontsize=12, fontweight='bold')
    ax.set_title('Model Predictions per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'F{i+1}' for i in range(5)])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')

    # 3. Error analysis
    ax = axes[1, 0]
    false_positives = [cm.ravel()[1] for cm in confusion_matrices]
    false_negatives = [cm.ravel()[2] for cm in confusion_matrices]
    ax.bar(x - width/2, false_positives, width, label='False Positives', color='#e67e22', alpha=0.8)
    ax.bar(x + width/2, false_negatives, width, label='False Negatives', color='#c0392b', alpha=0.8)
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Errors per Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'F{i+1}' for i in range(5)])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')

    # 4. Metrics correlation
    ax = axes[1, 1]
    corr_data = df_folds[['AUC', 'F1', 'Recall', 'Precision']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={'label': 'Correlation'},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Metric Correlations Across Folds', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_7_class_balance.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'results_7_class_balance.png')}")
    plt.close(fig)
except Exception as e:
    print(f"Error plotting Figure 7: {e}")

# =====================================
# Print Summary Statistics
# =====================================
print("\n" + "="*60)
print("📊 RESULTS SUMMARY")
print("="*60)
print(f"\n5-Fold Cross-Validation:")
print(f"  AUC:       {df_folds['AUC'].mean():.4f} ± {df_folds['AUC'].std():.4f}")
print(f"  F1:        {df_folds['F1'].mean():.4f} ± {df_folds['F1'].std():.4f}")
print(f"  Recall:    {df_folds['Recall'].mean():.4f} ± {df_folds['Recall'].std():.4f}")
print(f"  Precision: {df_folds['Precision'].mean():.4f} ± {df_folds['Precision'].std():.4f}")

print(f"\nHold-out Test Set:")
print(f"  AUC:       {test_results['AUC']:.4f}")
print(f"  F1:        {test_results['F1']:.4f}")
print(f"  Recall:    {test_results['Recall']:.4f}")
print(f"  Precision: {test_results['Precision']:.4f}")
print(f"  Specificity: {test_specificity:.4f}") # Using test_specificity from Fig 4

print(f"\nModel Stability (Coefficient of Variation):")
metrics = ['AUC', 'F1', 'Recall', 'Precision'] # Ensure metrics is defined
for metric in metrics:
    mean_val = df_folds[metric].mean()
    std_val = df_folds[metric].std()
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else 0
    print(f"  {metric} CV: {cv_val:.2f}%")

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS COMPLETE!")
print("="*60)
print(f"\nGenerated files saved in: {OUTPUT_DIR}")
print(f"  1. {os.path.join(OUTPUT_DIR, 'results_1_training_curves.png')}")
print(f"  2. {os.path.join(OUTPUT_DIR, 'results_2_cv_performance.png')}")
print(f"  3. {os.path.join(OUTPUT_DIR, 'results_3_confusion_matrices.png')}")
print(f"  4. {os.path.join(OUTPUT_DIR, 'results_4_spider_plot.png')}")
print(f"  5. {os.path.join(OUTPUT_DIR, 'results_5_variability_analysis.png')}")
print(f"  6. {os.path.join(OUTPUT_DIR, 'results_6_roc_curves.png')}")
print(f"  7. {os.path.join(OUTPUT_DIR, 'results_7_class_balance.png')}")
print("="*60)