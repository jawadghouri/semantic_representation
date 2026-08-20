#!/usr/bin/env python3
"""
Master execution script for semantic representation analysis pipeline.

Runs all visualization and analysis modules in sequence:
1. Data validation and loading
2. Heatmap generation (model-centric)
3. 2D PCA plots (paragraph-centric, Option A)
4. 3D PCA plots (paragraph-centric, Option A)
5. Prompt-response distance bar charts

Usage:
    python run_all_analysis.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.groups import DATA_DIR, OUTPUT_DIR, GROUP_CONFIG, EMBEDDING_MODELS


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted subsection header."""
    print(f"\n→ {title}")
    print("-" * 80)


def validate_environment():
    """Check that required packages and directories are in place."""
    print_header("ENVIRONMENT VALIDATION")

    try:
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn
        print("✓ All required packages available")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        sys.exit(1)

    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"⚠ Data directory not found: {DATA_DIR}")
        print(f"  Creating: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)
    else:
        n_files = len(list(data_path.glob("*.npy")))
        print(f"✓ Data directory found with {n_files} embedding files")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")

    print(f"\n✓ Configuration:")
    print(f"  • Groups: {len(GROUP_CONFIG)} semantic groups")
    print(f"  • Models: {', '.join(EMBEDDING_MODELS)}")
    total_ids = sum(len(g["ids"]) for g in GROUP_CONFIG)
    print(f"  • Total responses: {total_ids}")


def run_heatmap_analysis():
    """Generate distance heatmaps."""
    print_section("STEP 1: Heatmap Analysis (Model-Centric)")

    try:
        from visualization.heatmap_plots import plot_all_models

        print("Generating unnormalized heatmaps...")
        plot_all_models(normalized=False)

        print("\nGenerating normalized heatmaps...")
        plot_all_models(normalized=True)

        print("✓ Heatmap analysis complete")
        return True

    except Exception as e:
        print(f"✗ Heatmap analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pca_2d_analysis():
    """Generate 2D PCA scatter plots."""
    print_section("STEP 2: 2D PCA Plots (Paragraph-Centric, Option A)")

    try:
        from visualization.pca_plots_2d import plot_paragraph_centric_2d

        print("Generating unnormalized 2D PCA plots...")
        plot_paragraph_centric_2d(normalized=False)

        print("\nGenerating normalized 2D PCA plots...")
        plot_paragraph_centric_2d(normalized=True)

        print("✓ 2D PCA analysis complete")
        return True

    except Exception as e:
        print(f"✗ 2D PCA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pca_3d_analysis():
    """Generate 3D PCA scatter plots."""
    print_section("STEP 3: 3D PCA Plots (Paragraph-Centric, Option A)")

    try:
        from visualization.pca_plots_3d import plot_paragraph_centric_3d

        print("Generating unnormalized 3D PCA plots...")
        plot_paragraph_centric_3d(normalized=False)

        print("\nGenerating normalized 3D PCA plots...")
        plot_paragraph_centric_3d(normalized=True)

        print("✓ 3D PCA analysis complete")
        return True

    except Exception as e:
        print(f"✗ 3D PCA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_bar_chart_analysis():
    """Generate prompt-response distance bar charts."""
    print_section("STEP 4: Prompt-Response Distance Bar Charts")

    try:
        from visualization.bar_charts import plot_all_combinations

        print("Generating unnormalized bar charts...")
        plot_all_combinations(normalized=False)

        print("\nGenerating normalized bar charts...")
        plot_all_combinations(normalized=True)

        print("✓ Bar chart analysis complete")
        return True

    except Exception as e:
        print(f"✗ Bar chart analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: dict):
    """Print execution summary."""
    print_header("EXECUTION SUMMARY")

    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}  {step}")

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}/")
    print(f"\nNext steps:")
    print(f"  1. Review generated PNG files in {OUTPUT_DIR}/")
    print(f"  2. Update config/groups.py if needed for different groupings")
    print(f"  3. Re-run analysis pipeline: python run_all_analysis.py")


def main():
    """Execute the complete analysis pipeline."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  SEMANTIC REPRESENTATION ANALYSIS PIPELINE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    validate_environment()

    results = {
        "Heatmap Analysis": run_heatmap_analysis(),
        "2D PCA Plots": run_pca_2d_analysis(),
        "3D PCA Plots": run_pca_3d_analysis(),
        "Bar Chart Analysis": run_bar_chart_analysis(),
    }

    print_summary(results)

    all_success = all(results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
