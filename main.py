# main.py
import os
import datetime
from pathlib import Path
from tools.TemplateLogGenerator import TemplateLogGenerator
from tools.TemplateFactory import ImprovedTemplateFactory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-nl', '--num_logs', type=int, default=1000) #Android Log num 1487126
parser.add_argument('-nt', '--num_templates', type=int, default=10)
parser.add_argument('-ar', '--anomaly_rate', type=float, default=0.2)
parser.add_argument('-pf', '--param_log_file', type=str, default="")
parser.add_argument('-ip', '--inject_param_log', action='store_true')

args = parser.parse_args()

def main():
    # generated_logs ディレクトリが存在しない場合は作成
    logs_dir = Path("./generated_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ImprovedTemplateFactory からテンプレートとラベル、および頻度を取得
    factory = ImprovedTemplateFactory(num_templates=args.num_templates, anomaly_rate=args.anomaly_rate)  # 異常ラベルの割合を20%に設定
    templates_with_labels, frequencies = factory.generate_templates()

    # TemplateLogGenerator を初期化してログを生成
    log_generator = TemplateLogGenerator(
        templates_with_labels=templates_with_labels,
        template_frequencies=frequencies
    )
    if args.inject_param_log:
        logs = log_generator.generate_logs_inject_param(args.num_logs, args.param_log_file)
    else:
        logs = log_generator.generate_logs(args.num_logs)


    # ログファイルの名前を現在の日時で定義
    filename = logs_dir / f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # ログをファイルに書き込み
    with open(filename, 'w') as file:
        for log in logs:
            file.write(log + "\n")

    print(f"Logs have been generated and saved to {filename}")


if __name__ == "__main__":
    main()