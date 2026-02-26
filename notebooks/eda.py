"""
Exploratory Data Analysis for Housing Prices Dataset
=====================================================
Generates statistical summaries, distribution plots,
correlation heatmaps, and feature analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Configuration ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Housing.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})


def load_data():
    """Load and return the housing dataset."""
    df = pd.read_csv(DATA_PATH)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    return df


def plot_price_distribution(df):
    """Plot the distribution of property prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['price'], bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--', label=f"Mean: {df['price'].mean():,.0f}")
    axes[0].axvline(df['price'].median(), color='green', linestyle='--', label=f"Median: {df['price'].median():,.0f}")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(df['price'], vert=True)
    axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'price_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n Saved: price_distribution.png")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric features."""
    # Encode binary columns for correlation
    df_encoded = df.copy()
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})
    df_encoded = pd.get_dummies(df_encoded, columns=['furnishingstatus'], drop_first=True)

    corr = df_encoded.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(" Saved: correlation_heatmap.png")

    # Print top correlations with price
    price_corr = corr['price'].drop('price').sort_values(ascending=False)
    print(f"\n Top Correlations with Price:\n{price_corr}")


def plot_feature_vs_price(df):
    """Plot key features vs price."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Area vs Price (scatter)
    axes[0, 0].scatter(df['area'], df['price'], alpha=0.5, color='#2196F3', s=20)
    axes[0, 0].set_title('Area vs Price', fontweight='bold')
    axes[0, 0].set_xlabel('Area (sq ft)')
    axes[0, 0].set_ylabel('Price')

    # Bedrooms vs Price (box)
    df.boxplot(column='price', by='bedrooms', ax=axes[0, 1])
    axes[0, 1].set_title('Bedrooms vs Price', fontweight='bold')
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price')

    # Bathrooms vs Price (box)
    df.boxplot(column='price', by='bathrooms', ax=axes[0, 2])
    axes[0, 2].set_title('Bathrooms vs Price', fontweight='bold')
    axes[0, 2].set_xlabel('Bathrooms')
    axes[0, 2].set_ylabel('Price')

    # Stories vs Price (box)
    df.boxplot(column='price', by='stories', ax=axes[1, 0])
    axes[1, 0].set_title('Stories vs Price', fontweight='bold')
    axes[1, 0].set_xlabel('Stories')
    axes[1, 0].set_ylabel('Price')

    # Parking vs Price (box)
    df.boxplot(column='price', by='parking', ax=axes[1, 1])
    axes[1, 1].set_title('Parking vs Price', fontweight='bold')
    axes[1, 1].set_xlabel('Parking Spots')
    axes[1, 1].set_ylabel('Price')

    # Furnishing Status vs Price (box)
    df.boxplot(column='price', by='furnishingstatus', ax=axes[1, 2])
    axes[1, 2].set_title('Furnishing Status vs Price', fontweight='bold')
    axes[1, 2].set_xlabel('Furnishing Status')
    axes[1, 2].set_ylabel('Price')

    plt.suptitle('')  # Remove auto-generated suptitle
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_vs_price.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(" Saved: feature_vs_price.png")


def plot_categorical_analysis(df):
    """Analyze categorical features impact on price."""
    cat_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, feat in enumerate(cat_features):
        means = df.groupby(feat)['price'].mean()
        colors = ['#4CAF50' if v == 'yes' else '#F44336' for v in means.index]
        axes_flat[i].bar(means.index, means.values, color=colors, edgecolor='white')
        axes_flat[i].set_title(f'{feat.title()} vs Avg Price', fontweight='bold')
        axes_flat[i].set_ylabel('Average Price')
        for j, (idx, val) in enumerate(means.items()):
            axes_flat[i].text(j, val + val * 0.01, f'{val:,.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'categorical_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(" Saved: categorical_analysis.png")


if __name__ == '__main__':
    print("Housing Prices - Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_price(df)
    plot_categorical_analysis(df)

    print("\n" + "=" * 60)
    print("EDA Complete! Plots saved to data/eda_plots/")
    print("=" * 60)
