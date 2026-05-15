import sys
import os

# Add project root to path so all imports resolve
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    from detector.modern_adas_hud import run_adas
    run_adas(0)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nPress ENTER to close...")
    input()