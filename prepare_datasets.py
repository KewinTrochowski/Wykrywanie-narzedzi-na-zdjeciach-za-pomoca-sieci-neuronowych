import json
from sklearn.model_selection import train_test_split

# Wczytanie danych
with open('data.json', 'r') as f:
    data = json.load(f)

# Podział danych
train_val, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 20% na test
train_data, val_data = train_test_split(train_val, test_size=0.25, random_state=42)  # 25% z pozostałych na walidację

# Zapisanie podzielonych danych
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)
with open('val_data.json', 'w') as f:
    json.dump(val_data, f)
with open('test_data.json', 'w') as f:
    json.dump(test_data, f)
