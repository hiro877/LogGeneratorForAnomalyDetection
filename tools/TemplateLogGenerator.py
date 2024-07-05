import random
import string
import datetime
import sys

import numpy as np
from tqdm import tqdm
class TemplateLogGenerator:
    def __init__(self, pid="2227", tid="2227", level="D", component="TextView",
                 update_time_n=1, update_date_n=1, templates_with_labels=None, template_frequencies=None):
        """
        テンプレートベースのログジェネレーターを初期化します。

        Args:
        - pid: プロセスID。
        - tid: スレッドID。
        - level: ログレベル。
        - component: コンポーネント名。
        - update_time_n: 時間更新の頻度。
        - update_date_n: 日付更新の頻度。
        - templates_with_labels: テンプレートと異常ラベルのリスト。
        - template_frequencies: 各テンプレートの出現頻度。
        """
        self.pid = pid
        self.tid = tid
        self.level = level
        self.component = component
        self.update_time_n = update_time_n
        self.update_date_n = update_date_n
        self.current_time = datetime.datetime.now()
        self.last_time_update = 0
        self.last_date_update = 0
        self.templates = [tpl[0] for tpl in templates_with_labels] if templates_with_labels else []
        self.labels = {tpl[0]: tpl[1] for tpl in templates_with_labels} if templates_with_labels else {}
        self.template_frequencies = template_frequencies if template_frequencies else np.ones(len(self.templates))
        self.template_frequencies /= np.sum(self.template_frequencies)  # 頻度を正規化

    def random_string(self, length):
        """固定長のランダムな英数字文字列を生成します。"""
        letters_and_digits = string.ascii_letters + string.digits
        return ''.join(random.choice(letters_and_digits) for _ in range(length))

    def generate_param(self, param_type='either'):
        """パラメータの型に基づいてランダムなパラメータを生成します。"""
        if param_type == 'int':
            return str(random.randint(100, 999))
        elif param_type == 'string':
            return self.random_string(random.randint(5, 10))
        else:
            if random.choice([True, False]):
                return str(random.randint(100, 999))
            else:
                return self.random_string(random.randint(5, 10))

    def generate_log(self):
        """単一のログエントリを生成します。"""
        if self.last_time_update % self.update_time_n == 0:
            self.time = (self.current_time + datetime.timedelta(seconds=self.last_time_update)).strftime("%H:%M:%S.%f")[:-3]
        if self.last_date_update % self.update_date_n == 0:
            self.date = (self.current_time + datetime.timedelta(seconds=self.last_date_update)).strftime("%m-%d")

        index = np.random.choice(len(self.templates), p=self.template_frequencies)
        template = self.templates[index]
        label = self.labels[template]
        param = self.generate_param()
        content = template.replace("[param]", param)
        log = f"{label} {self.date} {self.time}  {self.pid}  {self.tid} {self.level} {self.component}: {content}"

        self.last_time_update += 1
        self.last_date_update += 1

        return log

    def generate_logs(self, n_lines=100):
        """複数のログエントリを生成します。"""
        return [self.generate_log() for _ in tqdm(range(n_lines))]


    def generate_log_inject_param(self, param_template_index, param_logs):
        """単一のログエントリを生成します。"""
        if self.last_time_update % self.update_time_n == 0:
            self.time = (self.current_time + datetime.timedelta(seconds=self.last_time_update)).strftime("%H:%M:%S.%f")[:-3]
        if self.last_date_update % self.update_date_n == 0:
            self.date = (self.current_time + datetime.timedelta(seconds=self.last_date_update)).strftime("%m-%d")

        index = np.random.choice(len(self.templates), p=self.template_frequencies)

        if index == param_template_index:
            # print(param_logs)
            line = param_logs.pop(0)
            # print("line", line)
            line = line.rstrip('\n').split(" ")
            # print("line2", line)
            label = line[-1]
            # print("label", label)
            # content = " ".join(line[:-2])
            content = " ".join(line[:3])
            # print("content", content)
            # sys.exit()
        else:
            template = self.templates[index]
            label = self.labels[template]
            param = self.generate_param()
            content = template.replace("[param]", param)

        log = f"{label} {self.date} {self.time}  {self.pid}  {self.tid} {self.level} {self.component}: {content}"

        self.last_time_update += 1
        self.last_date_update += 1

        return log

    def generate_logs_inject_param(self, n_lines=100, param_log_file=""):
        """複数のログエントリを生成します。"""
        param_template_index = -1
        if param_log_file != "":
            with open(param_log_file, encoding='utf-8') as f:
                param_logs = f.readlines()
            # print(param_logs)
            param_template_index = np.random.choice(len(self.templates), p=self.template_frequencies)
            return [self.generate_log_inject_param(param_template_index, param_logs) for _ in tqdm(range(n_lines))]
        else:
            print("ERROR: param_log_file in none")
            sys.exit()

    # def generate_mimic_logs_inject_param(self, n_lines=100, param_log_file=""):
    #     """複数のログエントリを生成します。"""
    #     param_template_index = -1
    #     if param_log_file != "":
    #         with open(param_log_file, encoding='utf-8') as f:
    #             param_logs = f.readlines()
    #         param_template_index = np.random.choice(len(self.templates), p=self.template_frequencies)
    #     return [self.generate_log_inject_param(param_template_index, param_logs) for _ in range(n_lines)]