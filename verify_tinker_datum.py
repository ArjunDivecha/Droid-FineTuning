
import sys
import os
import logging
from typing import Dict, Any

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# Mock environment variable if needed
os.environ['TINKER_API_KEY'] = "dummy_key"

try:
    from backend.tinker_client import TinkerTrainingClient
    from tinker import types
except ImportError as e:
    print(f"ImportError: {e}")
    print("Make sure you are running this from the project root.")
    sys.exit(1)

# Mock tokenizer
class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Return dummy tokens
        return [1, 2, 3] if "User" in text else [4, 5, 6]

def test_datum_construction():
    print("Testing Datum construction...")
    client = TinkerTrainingClient(api_key="dummy")
    tokenizer = MockTokenizer()
    
    conversation = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"}
        ]
    }
    
    try:
        datum = client._manual_process_conversation(conversation, tokenizer)
        print("Successfully created Datum:")
        print(datum)
        
        # Verify fields
        if not hasattr(datum, 'model_input'):
            print("❌ FAIL: Datum missing 'model_input'")
            return False
        if not hasattr(datum, 'loss_fn_inputs'):
            print("❌ FAIL: Datum missing 'loss_fn_inputs'")
            return False
            
        print("✅ PASS: Datum has correct structure")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during construction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_datum_construction()
    sys.exit(0 if success else 1)
