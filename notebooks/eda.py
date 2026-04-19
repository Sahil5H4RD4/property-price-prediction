"""
Exploratory Data Analysis for Housing Prices Dataset
=====================================================
Generates statistical summaries, distribution plots,
correlation heatmaps, and feature analysis.

Run directly:
    python notebooks/eda.py
"""

import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Housing.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'eda_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINARY_COLS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})


def load_data() -> pd.DataFrame:
    """Load and print a summary of the housing dataset."""
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded dataset: %d rows, %d columns", *df.shape)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    missing = df.isnull().sum()
    print(f"\nMissing Values:\n{missing[missing > 0] if missing.sum() else 'None'}")
    return df


def _save(filename: str) -> None:
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved %s", filename)


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram and box plot of property price distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['price'], bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--',
                    label=f"Mean: {df['price'].mean():,.0f}")
    axes[0].axvline(df['price'].median(), color='green', linestyle='--',
                    label=f"Median: {df['price'].median():,.0f}")
    axes[0].legend()

    axes[1].boxplot(df['price'], vert=True)
    axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price')

    plt.tight_layout()
    _save('price_distribution.png')


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap across all encoded features."""
    df_enc = df.copy()
    for col in BINARY_COLS:
        df_enc[col] = df_enc[col].map({'yes': 1, 'no': 0})
    df_enc = pd.get_dummies(df_enc, columns=['furnishingstatus'], drop_first=True)

    corr = df_enc.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save('correlation_heatmap.png')

    price_corr = corr['price'].drop('price').sort_values(ascending=False)
    print(f"\nTop Correlations with Price:\n{price_corr}")


def plot_feature_vs_price(df: pd.DataFrame) -> None:
    """Scatter and box plots for key numeric features vs price."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    scatter_cfg = dict(alpha=0.5, color='#2196F3', s=20)
    axes[0, 0].scatter(df['area'], df['price'], **scatter_cfg)
    axes[0, 0].set_title('Area vs Price', fontweight='bold')
    axes[0, 0].set_xlabel('Area (sq ft)')
    axes[0, 0].set_ylabel('Price')

    for ax, col, title in [
        (axes[0, 1], 'bedrooms', 'Bedrooms vs Price'),
        (axes[0, 2], 'bathrooms', 'Bathrooms vs Price'),
        (axes[1, 0], 'stories', 'Stories vs Price'),
        (axes[1, 1], 'parking', 'Parking vs Price'),
        (axes[1, 2], 'furnishingstatus', 'Furnishing Status vs Price'),
    ]:
        df.boxplot(column='price', by=col, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(col.title())
        ax.set_ylabel('Price')

    plt.suptitle('')
    plt.tight_layout()
    _save('feature_vs_price.png')


def plot_categorical_analysis(df: pd.DataFrame) -> None:
    """Bar chart of average price by each binary amenity feature."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, feat in enumerate(BINARY_COLS):
        means = df.groupby(feat)['price'].mean()
        colors = ['#4CAF50' if v == 'yes' else '#F44336' for v in means.index]
        axes_flat[i].bar(means.index, means.values, color=colors, edgecolor='white')
        axes_flat[i].set_title(f'{feat.title()} vs Avg Price', fontweight='bold')
        axes_flat[i].set_ylabel('Average Price')
        for j, (_, val) in enumerate(means.items()):
            axes_flat[i].text(j, val * 1.01, f'{val:,.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    _save('categorical_analysis.png')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Housing Prices — Exploratory Data Analysis")
    print("=" * 60)
    df = load_data()
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_price(df)
    plot_categorical_analysis(df)
    print("\nEDA complete — plots saved to data/eda_plots/")
