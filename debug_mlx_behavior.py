
import mlx.core as mx
from mlx.utils import tree_flatten

def test_behavior():
    # Test tree_flatten
    d = {"a": mx.array([1]), "b": mx.array([2])}
    
    print("Testing tree_flatten(d, destination={})...")
    flat_dict = tree_flatten(d, destination={})
    print(f"Type: {type(flat_dict)}")
    print(f"Value: {flat_dict}")
    
    print("\nTesting tree_flatten(d)...")
    flat_list = tree_flatten(d)
    print(f"Type: {type(flat_list)}")
    print(f"Value: {flat_list}")
    
    # Test mx.eval with dict
    print("\nTesting mx.eval(dict)...")
    try:
        mx.eval(d)
        print("mx.eval(dict) returned (might have done nothing)")
    except Exception as e:
        print(f"mx.eval(dict) failed: {e}")

    # Test mx.eval with list
    print("\nTesting mx.eval(list)...")
    try:
        mx.eval(flat_list)
        print("mx.eval(list) returned (might have done nothing)")
    except Exception as e:
        print(f"mx.eval(list) failed: {e}")
        
    # Test mx.eval with unpacked list
    print("\nTesting mx.eval(*list)...")
    try:
        mx.eval(*flat_list)
        print("mx.eval(*list) succeeded")
    except Exception as e:
        print(f"mx.eval(*list) failed: {e}")

if __name__ == "__main__":
    test_behavior()
