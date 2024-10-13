import json
import os
from copy import deepcopy
from utils import load_json, save_json, load
from dataset import HarryDataset


def get_character(data, character_att, character_rel):
    for session in data:
        names_att = data[session]["attributes"].keys()
        names_rel = data[session]["relations with Harry"].keys()
        # speakers = data[session]["speakers"]
        for name in names_att:
            full_name = data[session]["attributes"][name]["name"].strip()
            if full_name not in character_att:
                character_att[full_name] = {}
        for name in names_rel:
            full_name = data[session]["relations with Harry"][name]["name"].strip()
            if full_name not in character_rel:
                character_rel[full_name] = {}
    return character_att, character_rel


def run_get_character(path = "./data"):

    test_file = os.path.join(path, "en_test_set.json")
    train_file = os.path.join(path, "en_train_set.json")
    test_data = load_json(test_file)
    train_data = load_json(train_file)

    char_att_file = os.path.join(path, "character_att.json")
    char_rel_file = os.path.join(path, "character_rel.json")
    if os.path.exists(char_att_file) and os.path.exists(char_rel_file):
        character_att = load_json(char_att_file)
        character_rel = load_json(char_rel_file)
    else:
        character_att = {}
        character_rel = {}
        character_att, character_rel = get_character(test_data, character_att, character_rel)
        character_att, character_rel = get_character(train_data, character_att, character_rel)
        save_json(char_att_file, character_att)
        save_json(char_rel_file, character_rel)

    intersect = {name: {} for name in character_att.keys() if name in character_rel}
    # att_only = [name for name in character_att.keys() if name not in character_rel]
    # rel_only = [name for name in character_rel.keys() if name not in character_att]
    inter_file = os.path.join(path, "character_intersection.json")
    if not os.path.exists(inter_file):
        save_json(inter_file, intersect)
    print("Number of character attributes:", len(character_att))  # 96
    print("Number of characters in relations:", len(character_rel))  # 96


def filter_data(train_data, test_data, character):
    data_combined = {}
    idx = 0
    for data in [train_data, test_data]:

        for session, dat in data.items():
            temp_data = deepcopy(dat)
            names_att = list(dat["attributes"].keys())
            names_rel = list(dat["relations with Harry"].keys())
            names = list(set(names_att + names_rel))
            # temp_data["attributes"] = {}
            # temp_data["relations with Harry"] = {}
            for name in names:
                temp_data["attributes"] = {}
                temp_data["relations with Harry"] = {}
                if name in dat["attributes"]:
                    full_name = dat["attributes"][name]["name"].strip()
                    if full_name in character:
                        temp_data["attributes"][full_name] = deepcopy(dat["attributes"][name])
                if name in dat["relations with Harry"]:
                    full_name = dat["relations with Harry"][name]["name"].strip()
                    if full_name in character:
                        temp_data["relations with Harry"][full_name] = deepcopy(dat["relations with Harry"][name])
                        if len(temp_data["relations with Harry"]) == 1:
                            data_combined[idx] = deepcopy(temp_data)
                            idx += 1

                # if full_name in character:
                #     temp_data["attributes"][full_name] = deepcopy(dat["attributes"][name])
                #     temp_data["relations with Harry"][full_name] = deepcopy(dat["relations with Harry"][name])

            # if len(temp_data["relations with Harry"]) >0:
            #     data_combined[idx] = temp_data
            #     idx += 1

    return data_combined


def get_persona(character, path='./experiments/', aspect="all"):
    if aspect == "all" or aspect == "0":
        subpaths = ['entity', 'culture', 'religion', 'vocation', 'ideology', 'personality', 'subjectivity']
    else:
        subpaths = [aspect]
        # subpaths = ['subjectivity']  #['personality']  #['ideology']  # ['vocation'] #['religion']  # ['culture'] # ['entity']
    for subpath in subpaths:
        filepath = os.path.join(path, subpath, 'proposed.json')
        # with open(filepath, 'r') as f:
        data = load(filepath, name='test')
        # print("done")
        for dat in data['test']:
            character[dat['name']][subpath] = dat['label']

    return character


def get_dataset(path="./data"):
    new_data_name = "data_combined.json"
    character_file = os.path.join(path, "character_intersection.json")
    character = load_json(character_file)
    if os.path.exists(os.path.join(path,new_data_name)):
        data_combined = json.load(open(os.path.join(path, new_data_name)))
    else:
        test_file = os.path.join(path, "en_test_set.json")
        train_file = os.path.join(path, "en_train_set.json")
        test_data = load_json(test_file)
        train_data = load_json(train_file)
        data_combined = filter_data(train_data, test_data, character)
        with open(os.path.join(path, new_data_name), 'w') as f:
            json.dump(data_combined, f)

    return data_combined, character


if __name__ == "__main__":
    run_get_character()
    data_combined, character = get_dataset()
    character = get_persona(character)
    harry_data = HarryDataset(data_combined, character)

    for data in harry_data:
        print(data)
        exit()



    print("done!")


