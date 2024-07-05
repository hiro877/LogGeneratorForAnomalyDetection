import random
import numpy as np
from tqdm import tqdm
class ImprovedTemplateFactory:
    def __init__(self, num_templates=3, anomaly_rate=0.1, base_templates=None):
        """
        テンプレートとそれに対応する異常ラベルを生成するファクトリを初期化します。

        Args:
        - num_templates: 生成するテンプレートの数。
        - anomaly_rate: 異常ラベルを付けるテンプレートの割合。
        - base_templates: 基本となるテンプレートのリスト。未指定の場合は自動生成。
        """
        self.num_templates = num_templates
        self.anomaly_rate = anomaly_rate

        self.actions = [
            "Error", "Warning", "Info", "Access", "Update", "Check", "Delete",
            "Save", "Load", "Send", "Receive", "Initialize", "Shutdown"
        ]
        self.objects = [
            "file", "user", "data", "system", "configuration", "database",
            "message", "email", "account", "record", "transaction", "session", "cache"
        ]
        self.suffixes = [
            "failed", "completed", "started", "was accessed", "was updated",
            "was checked", "was deleted", "was saved", "was loaded",
            "was sent", "was received", "was initialized", "was shut down"
        ]
        self.additional_phrases = [
            "due to network error", "successfully", "with warnings", "with errors",
            "by admin", "by user", "automatically", "manually"
        ]

        if base_templates is None:
            self.base_templates = self.generate_unique_base_templates(num_templates)
        else:
            self.base_templates = base_templates



    def generate_unique_base_templates(self, num_templates=1000):
        base_templates = set()
        print("generate_unique_base_templates()... num_templates={}".format(num_templates))

        with tqdm(total=num_templates) as pbar:
            while len(base_templates) < num_templates:
                action = random.choice(self.actions)
                obj = random.choice(self.objects)
                suffix = random.choice(self.suffixes)
                additional = random.choice(self.additional_phrases)
                template = f"{action} {obj} [param] {suffix} {additional}"
                if template not in base_templates:
                    base_templates.add(template)
                    pbar.update(1)

        return list(base_templates)

    def generate_templates(self):
        """
        テンプレートとそれに対応する異常ラベルのリストを生成します。

        Returns:
        - テンプレートと異常ラベルのリストのタプル (list of (template, anomaly_label)).
        """
        print(len(self.base_templates), self.num_templates)
        selected_templates = random.sample(self.base_templates, self.num_templates)
        templates_with_labels = [
            (template, "ERR" if random.random() < self.anomaly_rate else "-")
            for template in selected_templates
        ]
        frequencies = np.random.rand(self.num_templates)
        frequencies /= np.sum(frequencies)  # 正規化
        return templates_with_labels, frequencies.tolist()

# 使用例
if __name__ == "__main__":
    factory = ImprovedTemplateFactory(num_templates=5, anomaly_rate=0.2)
    templates_with_labels, frequencies = factory.generate_templates()
    print(templates_with_labels, frequencies)
