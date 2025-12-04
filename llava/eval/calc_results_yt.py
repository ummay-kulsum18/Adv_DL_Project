import os
import json
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(data):
    labels = [item["label"] for item in data]
    predictions = [item["prediction"].lower().startswith("yes") for item in data]  # Convert to boolean
    #predictions = ["yes" in item["prediction"].lower() for item in data]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

data_root = 'results/yt_scam'
for target_source in ['crypto', 'giftcard', 'monetary']:
    data = []
    for file in glob.glob(f"{data_root}/outputs/*.json"):
        with open(file, "r") as f:
            x = json.load(f)
        if "label" not in x:
            continue
        #print(x)
        if x['source'] != target_source:
            continue
        data.append(x)
    print(target_source, len(data))
    metrics = calculate_metrics(data)
    print(metrics)


data = []
for file in glob.glob(f"{data_root}/outputs/*.json"):
    with open(file, "r") as f:
        x = json.load(f)
    if "label" not in x:
        continue
    data.append(x)
print(len(data))
metrics = calculate_metrics(data)
print(metrics)


with open(os.path.join(data_root, "results.json"), "w") as f:
    json.dump(metrics, f, indent=4)