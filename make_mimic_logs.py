# main.py
import os
import datetime
import sys
from pathlib import Path
from tools.TemplateLogGenerator import TemplateLogGenerator
from tools.TemplateFactory import ImprovedTemplateFactory
from tools.LogContentGenerator import LogContentGenerator
from tools.Tools import *
import argparse

"""
python make_mimic_logs.py -fref results/Android/Android_structured.freq

python make_mimic_logs.py -fref results/Android/Android_structured.freq -pf parameter_logs/generated_logs/dataset_train_40942.txt -ipa
"""

parser = argparse.ArgumentParser()
parser.add_argument('-nl', '--num_logs', type=int, default=-1) #Android Log num 1487126
parser.add_argument('-pf', '--param_log_file', type=str, default="")
parser.add_argument('-fref', '--frequencies_file', type=str, default="")
parser.add_argument('-ar', '--anomaly_rate', type=float, default=0.2)
parser.add_argument('-mp', '--masked_param', type=str, default="State")
parser.add_argument('-dt', '--dataset_type', type=str, default="train")
parser.add_argument('-ipa', "--inject_param_anomaly", action='store_true')

args = parser.parse_args()

def main():
    # generated_logs ディレクトリが存在しない場合は作成
    param_log_file=args.param_log_file
    logs_dir = Path("./generated_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    frequencies = load_frequencies_file(args.frequencies_file)
    if args.num_logs < 0:
        num_logs = int(frequencies.pop(0))
    else:
        num_logs = args.num_logs
        _ = int(frequencies.pop(0))

    max_freq = max(frequencies)
    # print(frequencies)
    # print(max_freq)
    # print(num_logs)
    # sys.exit()

    log_gen = LogContentGenerator()
    if param_log_file == "":
        param_log_file = log_gen.make_dataset(args.dataset_type, args.masked_param, int(0.01*max_freq*num_logs), args.anomaly_rate)
    # sys.exit()
    # ImprovedTemplateFactory からテンプレートとラベル、および頻度を取得
    factory = ImprovedTemplateFactory(num_templates=len(frequencies), anomaly_rate=args.anomaly_rate)  # 異常ラベルの割合を20%に設定
    # factory.generate_unique_base_templates(len(frequencies))
    templates_with_labels, _ = factory.generate_templates()

    # TemplateLogGenerator を初期化してログを生成
    log_generator = TemplateLogGenerator(
        templates_with_labels=templates_with_labels,
        template_frequencies=frequencies
    )

    if args.inject_param_anomaly:
        logs = log_generator.generate_logs_inject_param(num_logs, param_log_file)
    else:
        logs = log_generator.generate_logs(num_logs)

    # ログファイルの名前を現在の日時で定義
    filename = logs_dir / f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # ログをファイルに書き込み
    with open(filename, 'w') as file:
        for log in logs:
            file.write(log + "\n")

    print(f"Logs have been generated and saved to {filename}")


if __name__ == "__main__":
    main()