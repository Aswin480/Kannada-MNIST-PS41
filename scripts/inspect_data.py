import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def inspect_datasets():
    print("=== Loading datasets for deep inspection ===")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    dig_df = pd.read_csv('Dig-MNIST.csv')
    sub_df = pd.read_csv('sample_submission.csv')
    
    summary = {}
    
    # 1. Row counts & column counts
    datasets = {
        'train': train_df,
        'test': test_df,
        'Dig-MNIST': dig_df,
        'sample_submission': sub_df
    }
    
    for name, df in datasets.items():
        summary[name] = {
            'shape': list(df.shape),
            'n_samples': int(df.shape[0]),
            'n_columns': int(df.shape[1]),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
        }
        if 'label' in df.columns:
            summary[name]['label_column'] = 'label'
            summary[name]['classes'] = [int(c) for c in sorted(df['label'].unique())]
            summary[name]['class_counts'] = {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()}
        if 'id' in df.columns:
            summary[name]['id_column'] = 'id'
            summary[name]['id_min'] = int(df['id'].min())
            summary[name]['id_max'] = int(df['id'].max())
            
        pixel_cols = [c for c in df.columns if c.startswith('pixel')]
        if pixel_cols:
            pixels = df[pixel_cols].values
            summary[name]['pixel_count'] = len(pixel_cols)
            summary[name]['pixel_min'] = int(pixels.min())
            summary[name]['pixel_max'] = int(pixels.max())
            summary[name]['pixel_mean'] = float(pixels.mean())
            summary[name]['pixel_std'] = float(pixels.std())

    print("Inspection Summary:")
    print(json.dumps(summary, indent=2))
    
    with open('outputs/reports/data_inspection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved inspection summary to outputs/reports/data_inspection_summary.json")

    # Visualizations
    # 1. Genuine images from every digit class (0-9)
    kannada_digits = {
        0: 'Digit 0 (Sonne)',
        1: 'Digit 1 (Ondu)',
        2: 'Digit 2 (Eradu)',
        3: 'Digit 3 (Mooru)',
        4: 'Digit 4 (Naalku)',
        5: 'Digit 5 (Aidu)',
        6: 'Digit 6 (Aaru)',
        7: 'Digit 7 (Eylu)',
        8: 'Digit 8 (Entu)',
        9: 'Digit 9 (Ombattu)'
    }
    
    pixel_cols = [c for c in train_df.columns if c.startswith('pixel')]
    fig, axes = plt.subplots(10, 6, figsize=(12, 20))
    fig.suptitle('Kannada-MNIST Genuine Dataset Samples (Classes 0-9)', fontsize=16, fontweight='bold', y=0.995)
    
    for digit in range(10):
        digit_samples = train_df[train_df['label'] == digit][pixel_cols].values[:6]
        for col_idx in range(6):
            ax = axes[digit, col_idx]
            img = digit_samples[col_idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(kannada_digits[digit], fontsize=11, fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/eda_digits_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/plots/eda_digits_grid.png")

    # 2. Class distribution plot
    train_counts = train_df['label'].value_counts().sort_index()
    dig_counts = dig_df['label'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    width = 0.38
    
    rects1 = ax.bar(x - width/2, train_counts.values, width, label='Train.csv (Total: 60,000)', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, dig_counts.values, width, label='Dig-MNIST.csv (Total: 10,240)', color='#ff7f0e', edgecolor='black')
    
    ax.set_xlabel('Digit Class (0 - 9)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Frequency Distribution: train.csv vs. Dig-MNIST.csv', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {i}\n({kannada_digits[i]})" for i in range(10)], fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plt.savefig('outputs/plots/eda_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/plots/eda_class_distribution.png")

if __name__ == '__main__':
    inspect_datasets()
