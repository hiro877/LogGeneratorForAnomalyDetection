import argparse
import sys
import os
from statistics import mean, pstdev, median_low, median_high
from scipy.stats import kurtosis
import numpy as np
import math

# python investigate_analyzed_file.py --dataset Mac --data_dir ./results/Android/Mac
# python investigate_analyzed_file.py --dataset Linux --data_dir ./results/Android/Linux

parser = argparse.ArgumentParser()
##### Dataset params
parser.add_argument("--dataset", default="BGL", type=str)
parser.add_argument("--data_dir", default="./results/BGL", type=str)
parser.add_argument("--log_file", default="BGL.log", type=str)

def load_analyzed_file(params, log_file):
    with open(os.path.join(params["data_dir"], log_file)) as f:
        load_data = f.readlines()

    return load_data

def investigate_hist(load_data, file_name):
    results = []
    all_num=0
    for data in load_data[1:]:
        splited = data.split(" ")[-2]
        results.append(int(splited))
        all_num += int(splited)
    print("=" * 20)
    print("Result of Investigation {}".format(file_name))
    print("=" * 20)
    print("-Log Frequency")
    print(" : mean={}, pstdev={}".format(mean(results), pstdev(results)))
    print(" : median_low={}, median_high={}".format(median_low(results),
                                                                  median_high(results)))
    print(" : max={}, min={}".format(max(results), min(results)))
    # print("log_types      : {}".format(len(self.log_types)))
    # print("log_components : {}".format(len(self.log_components)))
    print("all_num      : {}".format((all_num)))
    ks = analyze_kurtosis_hist(results)
    print("Kurtosis     : {}".format(ks))
    gini_coeff, entropy, mad, mean_abs_dev = analyze_indicator_of_dispersion(results)
    print("gini_coeff   : {}".format(gini_coeff))
    print("entropy      : {}".format(entropy))
    print("mad          : {}".format(mad))
    print("mean_abs_dev : {}".format(mean_abs_dev))
    print("-" * 20)
    print("For Copy   : {} {} {} {} {} {} {} {} {} {} {} {}".format(mean(results), pstdev(results), median_low(results),
                                                    median_high(results), max(results), min(results), all_num, ks,
                                                        gini_coeff, entropy, mad, mean_abs_dev ))

def analyze_kurtosis_hist(hist_right):
    hist_left = hist_right[::-1]
    histogram = hist_left + hist_right
    return kurtosis(histogram)

def analyze_indicator_of_dispersion(data):
    # ジニ係数
    sorted_data = np.sort(data)
    n = len(data)
    gini_coeff = 2 * np.sum((np.arange(1, n + 1)) * sorted_data) / (n * np.sum(sorted_data)) - (n + 1) / n

    # エントロピー
    # 各ログの出現確率を計
    total_newest = sum(data)
    probabilities_newest = [f / total_newest for f in data]
    # エントロピーの計算
    entropy = -sum(p * math.log(p, 2) for p in probabilities_newest if p > 0)

    # MAD
    mad = np.median(np.abs(data - np.median(data)))

    # Mean Absolute Deviation
    mean_abs_dev = np.mean(np.abs(data - np.mean(data)))
    return gini_coeff, entropy, mad, mean_abs_dev

if __name__ == "__main__":
    params = vars(parser.parse_args())

    file_names = [".txt", "_structured.txt", "_windowed.txt", "_windowed_structured.txt"]
    for file_name in file_names:

        load_data = load_analyzed_file(params, params["dataset"]+file_name)
        # load_data = load_analyzed_file(params, "Thunderbird_5000000" + file_name)
        investigate_hist(load_data, params["dataset"]+file_name)