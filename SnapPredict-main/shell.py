import pickle

with open("output/models/test_data.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(len(data))
print(data)