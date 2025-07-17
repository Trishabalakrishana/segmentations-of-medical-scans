import sys
import traceback

# Add the src directory to the Python path
sys.path.append('src')

# ✅ Main function call from segmentation.py
from segmentation import run_classical_segmentation

if __name__ == "__main__":
    try:
        print("Starting classical segmentation pipeline...")
        results = run_classical_segmentation()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nResults saved in:")
        print("- outputs/grabcut/           # GrabCut segmentation")
        print("- outputs/graphcut/          # Graph Cut segmentation")
        print("- outputs/chanvese/          # Chan-Vese segmentation")
        print("- outputs/morphacwe/         # MorphACWE segmentation")
        print("- outputs/visualizations/    # Visual comparisons")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        traceback.print_exc()
        raise
