import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import yaml

with open("params.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

os.makedirs("data/tokenized", exist_ok=True)

dataset = pd.read_table(params["dataset_file"])
smiles = dataset['canonical_smiles']
smiles = np.array(smiles).reshape(-1)
idx = [i for i, x in enumerate(smiles) if len(x)<=params["max_len_smiles"]]
smiles = smiles[idx]
char_set = set()
for i in tqdm(range(len(smiles))):
    smiles[i] = smiles[i].ljust(params["max_len_smiles"])
    char_set = char_set.union(set(smiles[i]))
char_set_list = sorted(list(char_set))
char_to_int = dict((c, i) for i, c in enumerate(char_set))
int_to_char = dict((i, c) for i, c in enumerate(char_set))

with open("data/tokenized/char_to_int.json", 'w') as file:
    file.write(json.dumps(char_to_int))
with open("data/tokenized/int_to_char.json", 'w') as file:
    file.write(json.dumps(int_to_char))

smiles_int = np.zeros((len(smiles), 
    params["max_len_smiles"], 1), dtype=np.int32)
for i in tqdm(range(len(smiles))):
    for t, char in enumerate(smiles[i]):
        smiles_int[i, t, 0] = char_to_int[char]

x_train, x_test = train_test_split(smiles_int,
    test_size = 1 - params["pct_train"], random_state = 42)
x_validation, x_test = train_test_split(x_test, test_size =
    params["pct_test"]/(params["pct_test"]+params["pct_validation"]), 
    random_state = 42)

np.save("data/tokenized/x_train", x_train)
np.save("data/tokenized/x_validation", x_validation)
np.save("data/tokenized/x_test", x_test)

# smiles_tf = tf.data.Dataset.from_tensor_slices(smiles_int)
# smiles_tf = smiles_tf.shuffle(buffer_size=len(smiles_tf))

# train_size = int(config["pct_train"] * len(smiles_tf))
# validation_size = int(config["pct_validation"] * len(smiles_tf))
# test_size = int(config["pct_test"] * len(smiles_tf))

# x_train = smiles_tf.take(train_size)
# x_train_remaining = smiles_tf.skip(train_size)
# x_validation = x_train_remaining.take(validation_size)
# x_test = x_train_remaining.skip(validation_size)

# x_train.save("data/x_train.tfdata")
# x_validation.save("data/x_validation.tfdata")
# x_test.save("data/x_test.tfdata")
