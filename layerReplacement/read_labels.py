import json

def read_labels(dict_path):
    f = open(dict_path, "r")
    l = json.load(f)
    labels = {}
    for key in l:
        labels[int(key)] = l[key]
    return labels