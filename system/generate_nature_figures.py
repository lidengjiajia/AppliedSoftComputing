#!/usr/bin/env python3
"""
Nature-style figure generator for HierFed paper
Larger single-panel figures with elegant color palette
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch

# Nature-style configuration - LARGER fonts
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
})

# Elegant Nature-inspired color palette
COLORS = {
    'blue': '#4C72B0',      # Steel blue
    'orange': '#DD8452',    # Muted orange
    'green': '#55A868',     # Forest green
    'red': '#C44E52',       # Muted red
    'purple': '#8172B3',    # Soft purple
    'honest': '#4C72B0',
    'strategic': '#DD8452',
    'malicious': '#C44E52',
}

LINE_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#C44E52']

# LARGE single-panel figures
FIG_WIDTH = 5.5
FIG_HEIGHT = 3.5


def setup_axis(ax, xlabel=None, ylabel=None, ylim=None, yticks=None):
    """Apply consistent axis styling"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1.0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)


def fig1_method_comparison():
    """Method comparison - LARGE single panel"""
    methods = ['Median', 'Krum', 'M-Krum', 'FLTrust', 'HierFed']
    
    # Average across both datasets
    avg_scores = [0.690, 0.711, 0.732, 0.762, 0.845]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(methods))
    
    bars = ax.bar(x, avg_scores, width=0.6, color=LINE_COLORS, 
                  edgecolor='white', linewidth=1.0, alpha=0.9)
    
    # Highlight HierFed
    bars[-1].set_edgecolor('#2E4057')
    bars[-1].set_linewidth(2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    setup_axis(ax, ylabel='Average AUC Score', ylim=(0.60, 0.90), 
               yticks=[0.60, 0.70, 0.80, 0.90])
    
    plt.tight_layout()
    plt.savefig('fig_method_comparison.pdf', format='pdf')
    plt.close()
    print("Generated: fig_method_comparison.pdf")


def fig2_robustness_analysis():
    """Combined robustness analysis - Heterogeneity + Byzantine ratio in ONE figure"""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # X-axis: combined scenarios
    scenarios = ['α=0.1', 'α=0.5', 'α=1.0', 'β=20%', 'β=30%', 'β=40%']
    methods = ['Median', 'Krum', 'FLTrust', 'HierFed']
    
    # Data: heterogeneity (α) then Byzantine ratio (β)
    data = {
        'Median': [0.60, 0.66, 0.71, 0.72, 0.67, 0.62],
        'Krum': [0.62, 0.68, 0.73, 0.74, 0.70, 0.65],
        'FLTrust': [0.66, 0.72, 0.77, 0.76, 0.73, 0.68],
        'HierFed': [0.77, 0.81, 0.85, 0.85, 0.82, 0.80],
    }
    
    x = np.arange(len(scenarios))
    markers = ['o', 's', 'd', 'p']
    
    for i, method in enumerate(methods):
        lw = 2.5 if method == 'HierFed' else 1.5
        ms = 9 if method == 'HierFed' else 7
        ax.plot(x, data[method], marker=markers[i], markersize=ms, 
                linewidth=lw, label=method, color=LINE_COLORS[i], alpha=0.9)
    
    # Add vertical separator
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(1.0, 0.88, 'Heterogeneity', ha='center', fontsize=10, color='gray')
    ax.text(4.5, 0.88, 'Byzantine Ratio', ha='center', fontsize=10, color='gray')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    # Legend at top, horizontal
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=10)
    setup_axis(ax, ylabel='AUC Score', ylim=(0.55, 0.90), 
               yticks=[0.55, 0.65, 0.75, 0.85])
    
    plt.tight_layout()
    plt.savefig('fig_robustness_analysis.pdf', format='pdf')
    plt.close()
    print("Generated: fig_robustness_analysis.pdf")


def fig3_detection_performance():
    """Detection performance by attack type - SINGLE panel with grouped bars"""
    attacks = ['Sign-flip', 'Gaussian', 'ALIE', 'MinMax', 'Label-flip']
    
    # Average across both datasets
    precision = [0.67, 0.64, 0.46, 0.63, 0.57]
    recall = [0.96, 0.94, 0.70, 0.90, 0.92]
    f1 = [0.78, 0.76, 0.56, 0.74, 0.71]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(attacks))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', 
           color=COLORS['blue'], edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.bar(x, recall, width, label='Recall',
           color=COLORS['orange'], edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.bar(x + width, f1, width, label='F1',
           color=COLORS['green'], edgecolor='white', linewidth=0.8, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=11, rotation=15, ha='right')
    # Legend at top, horizontal
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=10)
    setup_axis(ax, ylabel='Score', ylim=(0.40, 1.05), 
               yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig('fig_detection_performance.pdf', format='pdf')
    plt.close()
    print("Generated: fig_detection_performance.pdf")


def fig4_reputation_evolution():
    """Reputation evolution - LARGE single panel"""
    rounds = list(range(0, 101, 10))
    
    honest = [0.50, 0.62, 0.71, 0.78, 0.83, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96]
    strategic = [0.50, 0.48, 0.45, 0.41, 0.38, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26]
    malicious = [0.50, 0.38, 0.28, 0.20, 0.14, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    ax.plot(rounds, honest, 'o-', color=COLORS['honest'], linewidth=2.5, 
            markersize=8, label='Honest clients', alpha=0.9)
    ax.plot(rounds, strategic, 's--', color=COLORS['strategic'], linewidth=2.5,
            markersize=8, label='Strategic attackers', alpha=0.9)
    ax.plot(rounds, malicious, '^:', color=COLORS['malicious'], linewidth=2.5,
            markersize=8, label='Persistent attackers', alpha=0.9)
    
    # Add threshold line
    ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(102, 0.30, 'Threshold', va='center', fontsize=9, color='gray')
    
    # Legend at top, horizontal
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=10)
    setup_axis(ax, xlabel='Communication Round', ylabel='Reputation Score', 
               ylim=(0, 1.0), yticks=[0, 0.25, 0.50, 0.75, 1.0])
    
    plt.tight_layout()
    plt.savefig('fig_reputation_evolution.pdf', format='pdf')
    plt.close()
    print("Generated: fig_reputation_evolution.pdf")


if __name__ == '__main__':
    print("Generating LARGE single-panel figures...")
    print("=" * 50)
    print(f"Figure size: {FIG_WIDTH} x {FIG_HEIGHT} inches")
    print("=" * 50)
    
    fig1_method_comparison()
    fig2_robustness_analysis()
    fig3_detection_performance()
    fig4_reputation_evolution()
    
    print("=" * 50)
    print("All figures generated!")
