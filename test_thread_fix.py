"""
Quick test to verify thread initialization fix works.
Tests that model can be created without RuntimeError about thread settings.
"""
import sys

print("Testing thread initialization fix...")

# Test 1: Import should not raise errors
print("\n1. Testing module import...")
try:
    from models.nPLSI import neuralPLSI
    print("   ✓ Module imported successfully")
except RuntimeError as e:
    if "interop_threads" in str(e):
        print(f"   ✗ Thread initialization error still present: {e}")
        sys.exit(1)
    raise
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create model on CPU
print("\n2. Testing model creation (CPU)...")
try:
    model = neuralPLSI(family='continuous')
    print("   ✓ Model created successfully")
    print(f"   ✓ Device: {model.device}")
except RuntimeError as e:
    if "interop_threads" in str(e):
        print(f"   ✗ Thread initialization error during model creation: {e}")
        sys.exit(1)
    raise
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Multiple model creation (should not error)
print("\n3. Testing multiple model creation...")
try:
    model2 = neuralPLSI(family='continuous')
    model3 = neuralPLSI(family='binary')
    print("   ✓ Multiple models created successfully")
except RuntimeError as e:
    if "interop_threads" in str(e):
        print(f"   ✗ Thread error on multiple model creation: {e}")
        sys.exit(1)
    raise

print("\n" + "="*60)
print("✓ ALL THREAD INITIALIZATION TESTS PASSED!")
print("="*60)
print("\nThe thread initialization error has been fixed:")
print("  • Thread settings are attempted at module import time")
print("  • Graceful fallback with try-except if already initialized")
print("  • Safe to create multiple models")
print("  • No RuntimeError about interop_threads")
