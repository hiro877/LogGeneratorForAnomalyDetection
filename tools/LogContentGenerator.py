import sys
from tqdm import tqdm
import numpy as np
import random
import os

class LogContentGenerator:
    color_dic = {"black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "end": "\033[0m"}

    def __init__(self, params=None, anomaly_params=None, out_dir="parameter_logs/generated_logs"):
        self.out_dir = out_dir

        if params is None:
            self.params = {
                "State": ['A', 'B', 'C'],
                "Value": {"A": (0, 150), "B": (80, 2500), "C": (500, 1500)},
                "Info": {"A": ['InfoA', 'InfoB'], "B": ['InfoD'], "C": ['InfoA', 'InfoC', 'InfoD']}
            }
        else:
            self.params = params

        if anomaly_params is None:
            self.anomaly_params = {
                "State": {"A": ['B', 'C'], "B": ['A', 'C'], "C": ['A', 'C']},
                "Value": {"A": (151, 10000), "B": [(0, 79), (2501, 10000)], "C": [(0, 499), (1501, 10000)]},
                "Info": {"A": ['InfoC', 'InfoD'], "B": ['InfoA', 'InfoB', 'InfoC'], "C": ['InfoB']}
            }
        else:
            self.anomaly_params = anomaly_params

    @staticmethod
    def print_color(text, color="red"):
        print(LogContentGenerator.color_dic[color] + text + LogContentGenerator.color_dic["end"])

    def generate_log_entries(self, anomaly_rate, num_entries, with_label=True):
        data = []
        print("generate_log_entries()...")
        for _ in tqdm(range(num_entries)):
            label = "-"
            rand_index = random.randint(0, len(self.params["State"]) - 1)
            state = self.params["State"][rand_index]
            value = self.params["Value"][state]
            value = np.random.randint(*value)
            info = np.random.choice(self.params["Info"][state])

            if np.random.rand() < anomaly_rate:
                state, value, info, _ = self.generate_anomaly_log_entries(state, value, info)
                label = "Anomaly"

            if with_label:
                data.append(f"state={state}, value={value}, {info} {label}")
            else:
                data.append(f"state={state}, value={value}, {info}")
        return data

    def generate_log_entries_test(self, anomaly_rate, num_entries, masked_param):
        # masked_param = ['State', 'Value', 'Info'][masked_params_index]
        print("generate_log_entries_test()...")
        data = []
        for _ in tqdm(range(num_entries)):
            label = "-"
            rand_index = random.randint(0, len(self.params["State"]) - 1)
            state = self.params["State"][rand_index]
            value = self.params["Value"][state]
            value = np.random.randint(*value)
            info = np.random.choice(self.params["Info"][state])

            if np.random.rand() < anomaly_rate:
                state, value, info, anomaly_param = self.generate_anomaly_log_entries(state, value, info, masked_param)
                label = "Anomaly"
            else:
                anomaly_param = "_"

            data.append(f"state={state}, value={value}, {info} {anomaly_param} {label}")
        return data

    def generate_anomaly_log_entries(self, state, value, info, anomaly_target=None):
        anomaly_param = ""
        if anomaly_target is None:
            anomaly_target = np.random.choice(['State', 'Value', 'Info'])

        if anomaly_target == "State":
            state = np.random.choice(self.anomaly_params["State"][state])
            anomaly_param = state
        elif anomaly_target == "Value":
            values = self.anomaly_params["Value"][state]
            if isinstance(values, list):
                rand_index = random.randint(0, len(values) - 1)
                values = values[rand_index]
            value = np.random.randint(*values)
            anomaly_param = value
        elif anomaly_target == "Info":
            info = np.random.choice(self.anomaly_params["Info"][state])
            anomaly_param = info
        else:
            self.print_color("generate_anomaly_log_entries(): anomaly_param is Incorrect")

        return state, value, info, anomaly_param

    def make_dataset(self, dataset_type, masked_param, log_num, anomaly_rate):
        """
        :param dataset_type:
        :param masked_param: ['State', 'Value', 'Info']
        :param log_num:
        :return:
        """
        print("make dataset... dataset_type={}".format(dataset_type))
        if dataset_type=="train":
            save_file_path = self.make_train_dataset(anomaly_rate, log_num)
        elif dataset_type=="test":
            save_file_path = self.make_test_dataset(anomaly_rate, log_num, masked_param)
        else:
            print("This dataset type is not available.")
            sys.exit()
        return save_file_path
    def make_train_dataset(self, anomaly_rate, log_num=1000):
        file_name = "dataset_train"
        os.makedirs(self.out_dir, exist_ok=True)
        # for num in range(log_num):
        logs = self.generate_log_entries(anomaly_rate, log_num)
        save_file_path = f"{self.out_dir}/{file_name}_{log_num}.txt"
        with open(save_file_path, 'w', encoding='utf-8') as f:
            for line in logs:
                f.write(line + '\n')
        return save_file_path

        # save_file_path = f"{self.out_dir}/{file_name}_{log_num}.txt"
        # # Generate a dataset of num entries
        # logs = generate_log_entries(log_num, anomaly_rate=0.1)
        #
        # # データをテキストファイルに書き出す
        # with open(save_file_path, 'w', encoding='utf-8') as f:
        #     for line in logs:
        #         f.write(line + '\n')
        # return save_file_path

    def make_test_dataset(self, anomaly_rate, log_num=10000, masked_param="State"):
        """

        :param log_num:
        :param log_type: ['State', 'Value', 'Info']
        :return:
        """
        log_type = ["state", "value", "info"]
        file_name = "dataset_test"
        os.makedirs(self.out_dir, exist_ok=True)
        logs = self.generate_log_entries_test(anomaly_rate, log_num, masked_param)
        save_file_path = f'{self.out_dir}/{file_name}_{type}.txt'
        with open(save_file_path, 'w', encoding='utf-8') as f:
            for line in logs:
                f.write(line + '\n')
        # for i, type in enumerate(log_type):
        #     logs = self.generate_log_entries_test(log_num, i)
        #
        #     with open(f'{self.out_dir}/{file_name}_{type}.txt', 'w', encoding='utf-8') as f:
        #         for line in logs:
        #             f.write(line + '\n')
        return save_file_path

if __name__ == '__main__':
    log_gen = LogContentGenerator()
    log_gen.make_train_dataset()
    log_gen.make_test_dataset()
