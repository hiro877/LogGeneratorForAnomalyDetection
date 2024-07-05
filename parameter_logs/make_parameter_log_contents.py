import numpy as np
import random
import os
color_dic = {"black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "end": "\033[0m"}

def print_color(text, color="red"):
    print(color_dic[color] + text + color_dic["end"])

def generate_log_entries(num_entries, anomaly_rate=0.1, params=None, anomaly_params=None, with_label=False):
    """
    For Train Data
    """
    if params is None:
        params = {
            "State": ['A', 'B', 'C'],
            "Value": {"A": (0, 150), "B": (80, 2500), "C": (500, 1500)},
            "Info": {"A": ['InfoA', 'InfoB'], "B": ['InfoD'], "C": ['InfoA', 'InfoC', 'InfoD']}
        }

    if anomaly_params is None:
        anomaly_params = {
            "State": {"A": ['B', 'C'], "B": ['A', 'C'], "C": ['A', 'C']},
            "Value": {"A": (151, 10000), "B": [(0, 79), (2501, 10000)], "C": [(0, 499), (1501, 10000)]},
            "Info": {"A": ['InfoC', 'InfoD'], "B": ['InfoA', 'InfoB', 'InfoC'], "C": ['InfoB']},
        }

    data = []
    for _ in range(num_entries):
        label = "-"
        rand_index = random.randint(0, len(params)-1)
        state = params["State"][rand_index]
        value = params["Value"][state]
        value = np.random.randint(*value)
        info = np.random.choice(params["Info"][state])

        if np.random.rand() < anomaly_rate:
            anomaly_param = np.random.choice(['State', 'Value', 'Info'])
            state, value, info, anomaly_param_ = generate_anomaly_log_entries(state, value, info, anomaly_param, anomaly_params)
            label = "Anomaly"
        if with_label:
            data.append(f"state={state}, value={value}, {info} {label}")
        else:
            data.append(f"state={state}, value={value}, {info}")
    return data

def generate_log_entries_test(num_entries, masked_params_index, anomaly_rate=0.1, params=None, anomaly_params=None):
    """
    For Train Data
    """

    masked_param = ['State', 'Value', 'Info'][masked_params_index]
    if params is None:
        params = {
            "State": ['A', 'B', 'C'],
            "Value": {"A": (0, 150), "B": (80, 2500), "C": (500, 1500)},
            "Info": {"A": ['InfoA', 'InfoB'], "B": ['InfoD'], "C": ['InfoA', 'InfoC', 'InfoD']}
        }

    if anomaly_params is None:
        anomaly_params = {
            "State": {"A": ['B', 'C'], "B": ['A', 'C'], "C": ['A', 'C']},
            "Value": {"A": (151, 10000), "B": [(0, 79), (2501, 10000)], "C": [(0, 499), (1501, 10000)]},
            "Info": {"A": ['InfoC', 'InfoD'], "B": ['InfoA', 'InfoB', 'InfoC'], "C": ['InfoB']},
        }

    data = []
    anomaly_param = "_"
    for _ in range(num_entries):
        label = "-"
        rand_index = random.randint(0, len(params)-1)
        state = params["State"][rand_index]
        value = params["Value"][state]
        value = np.random.randint(*value)
        info = np.random.choice(params["Info"][state])

        if np.random.rand() < anomaly_rate:
            state, value, info, anomaly_param = generate_anomaly_log_entries(state, value, info, masked_param, anomaly_params)
            label = "Anomaly"
        data.append(f"state={state}, value={value}, {info} {anomaly_param} {label}")

    return data

def generate_anomaly_log_entries(state, value, info, anomaly_target, anomaly_params):
    # print("Processing generate_anomaly_log_entries)~ ...")
    anomaly_param = ""
    if anomaly_target == "State":
        state = np.random.choice(anomaly_params["State"][state])
        anomaly_param = state
    elif anomaly_target == "Value":
        # value = np.random.randint(151, 500)  # Anomalous value for state A
        values = anomaly_params["Value"][state]
        if isinstance(values, list):
            rand_index = random.randint(0, len(values)-1)
            values = values[rand_index]
        value = np.random.randint(*values)
        anomaly_param = value
    elif anomaly_target == "Info":
        info = np.random.choice(anomaly_params["Info"][state])
        anomaly_param = info
    else:
        print_color("generate_anomaly_log_entries(): anomaly_param is Incorrect")

    return state, value, info, anomaly_param

def make_train_dataset(out_dir="generated_logs"):
    # log_num = [10000, 50000, 100000, 500000, 1000000]
    log_num = [1000]
    file_name = "dataset_train"
    os.makedirs(out_dir, exist_ok=True)
    for num in log_num:
        # Generate a dataset of num entries
        logs = generate_log_entries(num, anomaly_rate=0.1)

        # データをテキストファイルに書き出す
        with open("{}/{}_{}.txt".format(out_dir, file_name, num), 'w', encoding='utf-8') as f:
            for line in logs:
                f.write(line + '\n')

def make_test_dataset(out_dir="generated_logs"):
    log_type = ["state", "value", "info"]
    file_name = "dataset_test"
    masked_params_index = 1
    os.makedirs(out_dir, exist_ok=True)
    for i, type in enumerate(log_type):
        logs = generate_log_entries_test(10000, i, anomaly_rate=0.5)

        # データをテキストファイルに書き出す
        with open('{}/{}_{}.txt'.format(out_dir, file_name, type), 'w', encoding='utf-8') as f:
            for line in logs:
                f.write(line + '\n')
if __name__ == '__main__':
    make_train_dataset()
    make_test_dataset()
