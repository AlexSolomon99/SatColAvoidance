import json

path = r"E:\Alex\UniBuc\MasterThesis\src\recorded_models\policy_methods_models\policy_model_8_dir\model_conf.json"

nn_conf = {
    "init_layer": 6968,
    "hidden_layer_1": 1000,
    "hidden_layer_2": 100,
    "output_layer": 6
}

# Convert and write JSON object to file
with open(path, "w") as outfile:
    json.dump(nn_conf, outfile)
