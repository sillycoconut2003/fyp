import pickle

# Check RandomForest model structure
with open('models/RandomForest_model.pkl', 'rb') as f:
    rf_data = pickle.load(f)

print("RandomForest model keys:", list(rf_data.keys()))
print("\nSample of data:")
for key, value in rf_data.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
