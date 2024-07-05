import argparse
import sys
import os
from tools import LogAnalyzer

# sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
"""
-BGL
python analyze.py --use_template
- Android
python analyze.py --dataset Android --data_dir ./datasets/Android/Android_v1/ --log_file Android.log --use_template
- Thunderbird
python analyze.py --dataset Thunderbird --data_dir ./datasets/Thunderbird/ --log_file Thunderbird_5000000.log1aa --use_template
- Windows
python analyze.py --dataset Windows --data_dir ./datasets/Windows/ --log_file Windows.log.aa --use_template
- Linux
python analyze.py --dataset Linux --data_dir ./datasets/Linux/ --log_file Linux.log --use_template
- Mac
python analyze.py --dataset Mac --data_dir ./datasets/Mac/  --log_file Mac.log --use_template

"""


parser = argparse.ArgumentParser()
##### Dataset params
parser.add_argument("--dataset", default="BGL", type=str)
parser.add_argument("--data_dir", default="./datasets/BGL/", type=str)
parser.add_argument("--save_dir", default="./results/", type=str)
parser.add_argument("--preprocessed_dir", default=None, type=str)
parser.add_argument("--log_file", default="BGL.log", type=str)
parser.add_argument("--use_data_size", default=None, type=int)
parser.add_argument("--use_template", action='store_true')


"""" Analyze """
parser.add_argument("--analyze_adfuller", action='store_true')

if __name__ == "__main__":
    params = vars(parser.parse_args())
    print(params)
    print(params["dataset"])

    log_analyzer = LogAnalyzer(params["dataset"], params["data_dir"], params["save_dir"], params["preprocessed_dir"], params["log_file"], params["use_data_size"], params["use_template"], params["analyze_adfuller"])

    log_analyzer.analyze()
    print()
    log_analyzer.analyze_log_hist()
    print()
    log_analyzer.analyze_windowed_hist()