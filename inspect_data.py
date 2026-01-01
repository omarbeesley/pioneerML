
import numpy as np
import os

path = "ML/npy_data/hits_batch_0.npy"
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

print(f"Loading {path}...")
try:
    chunk = np.load(path, allow_pickle=True)
    print(f"Loaded chunk type: {type(chunk)}")
    print(f"Chunk shape/len: {len(chunk) if hasattr(chunk, '__len__') else 'scalar'}")
    
    if len(chunk) > 0:
        first_group = chunk[0]
        print(f"First group type: {type(first_group)}")
        if isinstance(first_group, np.ndarray):
            print(f"First group shape: {first_group.shape}")
            print(f"First group content (first 5 rows):\n{first_group[:5]}")
        else:
            print(f"First group: {first_group}")
            
    # Check how many are not None
    non_none = sum(1 for x in chunk if x is not None)
    print(f"Non-None items: {non_none}")
    
except Exception as e:
    print(f"Error: {e}")
