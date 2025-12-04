
import os
import json
import glob

data_root = '/mnt/bum/mmiemon/LLaVA-NeXT/results/llava_ov/VSI'

result_list = []
for file in glob.glob(f"{data_root}/outputs/*.json"):
    with open(file, "r") as f:
        result_list.append(json.load(f))

print(len(result_list))

with open("/mnt/bum/mmiemon/datasets/VSI/formatted_dataset.json", "r") as f:
    dataset = json.load(f)
question_types = {s['id']:s["question_type"] for s in dataset}

results = {"all": {"correct": 0, "total": 0}}
for sample in result_list:
    sample["question_type"] = question_types[sample['id']]
    if "object_rel_direction" in sample["question_type"]:
        sample["question_type"] = "object_rel_direction"
    results["all"]["total"] += 1
    if "question_type" in sample:
        if sample["question_type"] not in results:
            results[sample["question_type"]] = {"correct": 0, "total": 0}
        results[sample["question_type"]]["total"] += 1
        
    if sample["prediction"][0] == sample["answer"]:
        results["all"]["correct"] += 1
        if "question_type" in sample:
            results[sample["question_type"]]["correct"] += 1

for key in results:
    results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]

print(results)

with open(os.path.join(data_root, "results.json"), "w") as f:
    json.dump(results, f, indent=4)