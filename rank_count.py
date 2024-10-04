import json


with open('test_result.json', 'r') as f:
    data = json.load(f)

for value in data:
    print(len(value["ranks"]))
