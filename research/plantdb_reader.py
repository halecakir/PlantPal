import json
import os


ROOT_PATH = "plant-db"

common2latin = {}
latin2common = {}

total = 0
missing_common_name = 0
for file in os.listdir(ROOT_PATH):
    total += 1
    full_path = os.path.join(ROOT_PATH, file)
    try:
        with open(full_path) as target:
            json_f = json.load(target)
            latin = json_f["pid"].lower()
            common_name = json_f["common_name"].lower()
            common2latin[common_name] = latin
            latin2common[latin] = common_name
    except Exception as e:
        missing_common_name += 1
        #print(f"Unknown error {e}")
        



def return_latin_name(user_inp, c2l, l2c):
    if user_inp in c2l:
        return c2l[user_inp]
    elif user_inp in l2c:
        return user_inp
    else:
        return None

def read_plant_info_given_latin(latin_name):
    with open(f"{ROOT_PATH}/{latin_name}.json") as plant_file:
        content = json.load(plant_file)
    return print(content["maintenance"])

def partial_str_match(user_inp, c2l, l2c):
    list_of_matched_common_name = []
    list_of_matched_latin_name = []
    for key in c2l:
        if user_inp in key:
            list_of_matched_common_name.append(key)
    
    max_item = None
    max_item_type = None
    c_length = 0
    if list_of_matched_common_name:
        c_length =  len(max(list_of_matched_common_name))
        max_c_item = max(list_of_matched_common_name)
        max_item = max_c_item
        max_item_type = "common"

    for key in l2c:
        if user_inp in key:
            list_of_matched_latin_name.append(key)

    l_length = 0
    if list_of_matched_latin_name:
        l_length =  len(max(list_of_matched_latin_name))
        max_l_item = max(list_of_matched_latin_name)
        if l_length > c_length:
            max_item = max_l_item
            max_item_type = "latin"
        
    
    if max_item and max_item_type =="common":
        max_item  = c2l[max_item] 

    return max_item



# Scenarios 


# 1. Full name
user_inp = 'cashmere bouquet'
user_inp = user_inp.lower()

# latin = return_latin_name(user_inp, common2latin, latin2common)
# read_plant_info_given_latin(latin)

# 2. Corner cases
## 2.1 partial match
latin = partial_str_match("cala li", common2latin, latin2common)
print('Found', latin)