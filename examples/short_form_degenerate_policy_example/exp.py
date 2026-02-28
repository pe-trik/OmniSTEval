import json
with open("references.txt", "w") as f:
    for line in open("instances.log"):
        f.write(json.loads(line)['reference'] + "\n")