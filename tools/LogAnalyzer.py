"""
- Raw Log
1. 単位時間あたりのlog数(per 1s)
2. Type総数
3. Component総数
4. logの総数
5. 各logの頻度

- Template(BGL)
1. 単位時間あたりのTemplate数
2. Template総数
3. 各Templateの頻度

- 異常Log
1. 異常ログの数
2. NormalログとAnomalyログの比率
3. 異常ログの分布

- 定常性
"""
"""
- 改行コード変換
nkf -Lu foo.txt
- 文字コード変換
nkf -w --overwrite foo.tzt 
- Split Data

"""

import os
from logparser import Drain
import pandas as pd
from statistics import mean, pstdev, median_low, median_high
import statsmodels.tsa.api as tsa
import datetime
from scipy.stats import kurtosis
from collections import Counter
import concurrent.futures
class LogAnalyzer:
    def __init__(self, dataset, input_dir, save_dir, preprocessed_dir, log_file, use_data_size, use_template, analyze_adfuller):
        self.dataset_name = dataset
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.preprocessed_dir = preprocessed_dir
        self.log_file = log_file
        self.dataset_path = os.path.join(self.input_dir, self.log_file)
        self.use_data_size = use_data_size
        self.use_template = use_template
        self.analyze_adfuller = analyze_adfuller

        self.parser = None
        self.df_log = None
        self.log2id = {}
        # For windowed logs
        self.all_ids = []
        self.window_size = 10
        self.sliding_size = 1

        # self.df_log = None
        self.make_parser()
        self.load_data()

        # Raw Log
        self.lognum_per1s = []
        self.log_types = set()
        self.log_components = set()
        self.log_num_all = 0
        # Anomaly Raw Log
        self.anomaly_log_num_all = 0
        self.anomaly_indexes = []

        # Template
        self.templatenum_per1s = []

        # ていじょうせい
        self.ids = []
        self.binaries = []
        self.freqensies = []


    def f(self):
        return 'hello world'

    def make_parser(self):
        print("Make Parser using {}".format(self.dataset_name))
        output_dir = 'Drain_result/'  # The output directory of parsing results
        if self.dataset_name == "BGL":
            log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # HDFS log format
            regex = [r'core\.\d+']
            st = 0.5  # Similarity threshold
            depth = 4  # Depth of all leaf nodes
        if self.dataset_name == "Android":
            log_format = '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>'
            regex = [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
            st = 0.2  # Similarity threshold
            depth = 6  # Depth of all leaf nodes
        if self.dataset_name == "Thunderbird":
            log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
            regex = [r'(\d+\.){3}\d+']
            st = 0.5  # Similarity threshold
            depth = 4  # Depth of all leaf nodes
        if self.dataset_name == "Windows":
            log_format = '<Date> <Time>, <Level>                  <Component>    <Content>'
            regex = [r'0x.*?\s']
            st = 0.7  # Similarity threshold
            depth = 5  # Depth of all leaf nodes
        if self.dataset_name == "Linux":
            log_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
            regex = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
            st = 0.39  # Similarity threshold
            depth = 6  # Depth of all leaf nodes
        if self.dataset_name == "Mac":
            log_format = '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>'
            regex = [r'([\w-]+\.){2,}[\w-]+']
            st = 0.7  # Similarity threshold
            depth = 6  # Depth of all leaf nodes

        self.parser = Drain.LogParser(log_format, indir=self.input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        self.parser.logName = self.log_file

        self.parser.parse(self.log_file)
        if(self.use_template):
            if not os.path.isfile(self.parser.get_structed_psth()):
                print("parsing file ...")
                self.parser.parse(self.log_file)

    def load_data(self):
        print("Load Dataset {}".format(self.dataset_name))
        self.load_common_data()

    def load_common_data(self):
        if self.use_template:
            self.parser.df_log = pd.read_csv(self.parser.get_structed_psth(), engine="c", na_filter=False, memory_map=True)
        else:
            save_path = os.path.join(self.input_dir, "preprocessed")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "pandas_"+self.log_file.split(".")[0]+".pkl")
            print("save preprocessed path: ", save_path)
            if os.path.exists(save_path):
                print("save pandas file")
                self.parser.df_log = pd.read_pickle(save_path)  # 圧縮無し
                return

            if self.use_data_size:
                self.parser.load_data_limited(self.use_data_size)
            else:
                self.parser.load_data()

            self.parser.df_log.to_pickle(save_path)  # 圧縮無し

    def load_bgl_data(self):
        save_path = os.path.join(self.input_dir, "preprocessed")
        print(save_path)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "pandas_"+self.log_file.split(".")[0]+".pkl")
        print(save_path)
        if os.path.exists(save_path):
            print("save pandas file")
            self.parser.df_log = pd.read_pickle(save_path)  # 圧縮無し
            return

        self.parser.load_data()
        # self.df_log = self.parser.df_log
        self.parser.df_log.to_pickle(save_path)  # 圧縮無し

    def load_android_data(self):
        if self.preprocessed_dir:
            save_path = os.path.join(self.preprocessed_dir, "preprocessed")
        else:
            save_path = os.path.join(self.input_dir, "preprocessed")
        print(save_path)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "pandas_"+self.log_file.split(".")[0]+".pkl")
        print(save_path)
        if os.path.exists(save_path):
            print("save pandas file")
            self.parser.df_log = pd.read_pickle(save_path)  # 圧縮無し
            return

        if self.use_template:
            self.df_log = pd.read_csv(os.path.join(self.input_dir, self.log_file), engine="c", na_filter=False, memory_map=True)
        else:
            self.parser.load_data()
            self.parser.df_log.to_pickle(save_path)  # 圧縮無し

    def load_thunderbird_data(self):
        save_path = os.path.join(self.input_dir, "preprocessed")
        print(save_path)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "pandas_"+self.log_file.split(".")[0]+".pkl")
        print(save_path)
        if os.path.exists(save_path):
            print("save pandas file")
            self.parser.df_log = pd.read_pickle(save_path)  # 圧縮無し
            return

        self.parser.load_data()
        self.parser.df_log.to_pickle(save_path)  # 圧縮無し

    def analyze(self):
        if self.dataset_name == "BGL":
            self.analyze_bgl()
        if self.dataset_name == "Android":
            self.analyze_android()
        if self.dataset_name == "Thunderbird":
            self.analyze_thunderbird()
        if self.dataset_name == "Windows":
            self.analyze_windows()
        if self.dataset_name == "Linux":
            self.analyze_linux()
        if self.dataset_name == "Mac":
            self.analyze_mac()

    def analyze_bgl(self):
        current_time = self.parser.df_log["Timestamp"][0]
        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]
        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            timestamp = line["Timestamp"]
            component = line['Component']
            label = line['Label']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1



            if label != "-":
                self.anomaly_log_num_all += 1
                self.anomaly_indexes.append(idx)
                self.binaries.append(1)
            else:
                self.binaries.append(0)

            per1s += 1
            if timestamp != current_time:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=timestamp
        self.print_results()

    def analyze_android(self):
        s_format = '%H:%M:%S.%f'
        current_time = self.parser.df_log["Time"][0]
        current_time = datetime.datetime.strptime(current_time, s_format)

        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]

        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            time = datetime.datetime.strptime(line["Time"], s_format)
            component = line['Component']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1

            per1s += 1
            if time.second != current_time.second:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=time
        self.print_results()

    def analyze_thunderbird(self):
        current_time = self.parser.df_log["Timestamp"][0]
        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]
        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            timestamp = line["Timestamp"]
            component = line['Component']
            label = line['Label']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1



            if label != "-":
                self.anomaly_log_num_all += 1
                self.anomaly_indexes.append(idx)
                self.binaries.append(1)
            else:
                self.binaries.append(0)

            per1s += 1
            if timestamp != current_time:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=timestamp
        self.print_results()

    def analyze_linux(self):
        s_format = '%H:%M:%S'
        current_time = self.parser.df_log["Time"][0]
        current_time = datetime.datetime.strptime(current_time, s_format)
        print(current_time)
        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]
        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            time = datetime.datetime.strptime(line["Time"], s_format)
            component = line['Component']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1

            per1s += 1
            if time.second != current_time.second:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=time
        self.print_results()

    def analyze_windows(self):
        s_format = '%H:%M:%S'
        current_time = self.parser.df_log["Time"][0]
        current_time = datetime.datetime.strptime(current_time, s_format)
        print(current_time)
        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]
        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            try:
                time = datetime.datetime.strptime(line["Time"], s_format)
            except ValueError:
                time = current_time
            component = line['Component']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1

            per1s += 1
            if time.second != current_time.second:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=time
        self.print_results()

    def analyze_mac(self):
        s_format = '%H:%M:%S'
        current_time = self.parser.df_log["Time"][0]
        current_time = datetime.datetime.strptime(current_time, s_format)
        print(current_time)
        per1s= 0
        log_id = 0

        self.log_num_all = self.parser.df_log.shape[0]
        for idx, line in self.parser.df_log.iterrows():
            if self.use_template:
                content = line['EventId']
            else:
                content = line['Content']
            time = datetime.datetime.strptime(line["Time"], s_format)
            component = line['Component']

            self.log_types.add(content)
            self.log_components.add(component)
            if content in self.log2id:
                self.ids.append(self.log2id[content])
            else:
                self.ids.append(log_id)
                self.log2id[content] = log_id
                log_id += 1

            per1s += 1
            if time.second != current_time.second:
                self.lognum_per1s.append(per1s)
                per1s = 0
                current_time=time
        self.print_results()

    # def analyze_log_hist(self):
    #     # print("- ヒストグラム -")
    #     print("=" * 20)
    #     print("=== analyze_log_hist ===")
    #     print("=" * 20)
    #     save_path = os.path.join(self.save_dir, self.log_file.split(".")[0])
    #     os.makedirs(save_path, exist_ok=True)
    #     if self.use_template:
    #         save_path = os.path.join(save_path, self.log_file.split(".")[0] + "_structured.txt")
    #     else:
    #         save_path = os.path.join(save_path, self.log_file.split(".")[0]+".txt")
    #     results = {}
    #     all_num = 0
    #     for log in self.log_types:
    #         if self.use_template:
    #             df = (self.parser.df_log["EventId"] == log)
    #         else:
    #             df = (self.parser.df_log["Content"] == log)
    #         results[log] = df.sum()
    #         all_num += df.sum()
    #     results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    #     self.save_hist(save_path, results, all_num)
    #     results = [i[1] for i in results]
    #     self.analyze_kurtosis_hist(results)

    def analyze_windowed_hist(self):
        print("="*20)
        print("=== analyze_windowed ===")
        print("=" * 20)
        windowed_logs = []
        results = {}
        all_num = 0
        for i in range(0, len(self.ids) - self.window_size + 1, self.sliding_size):
            window = self.ids[i:i + self.window_size]
            # print(window)
            window = ', '.join(map(str, window))
            windowed_logs.append(window)
            if window in results:
                results[window] += 1
            else:
                results[window] = 1
            all_num+=1

        save_path = os.path.join(self.save_dir, self.log_file.split(".")[0])
        os.makedirs(save_path, exist_ok=True)
        if self.use_template:
            save_path = os.path.join(save_path, self.log_file.split(".")[0] + "_windowed_structured.txt")
        else:
            save_path = os.path.join(save_path, self.log_file.split(".")[0]+"_windowed.txt")
        results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.save_hist(save_path, results, all_num)
        results = [i[1] for i in results]
        self.analyze_kurtosis_hist(results)


    def print_results(self):
        print("="*20)
        print("Result of Analysing {}".format(self.dataset_name))
        print("=" * 20)
        print("- Raw Log")
        print("lognum_per1s   : mean={}, pstdev={}".format(mean(self.lognum_per1s), pstdev(self.lognum_per1s)))
        print("               : median_low={}, median_high={}".format(median_low(self.lognum_per1s), median_high(self.lognum_per1s)))
        print("               : max={}, min={}".format(max(self.lognum_per1s), min(self.lognum_per1s)))
        print("log_types      : {}".format(len(self.log_types)))
        print("log_components : {}".format(len(self.log_components)))
        print("log_num_all    : {}".format(self.log_num_all))
        print("- Anomaly Raw Log")
        print("anomaly_log_num_all : {}".format(self.anomaly_log_num_all))

        if self.analyze_adfuller:
            # print("- ADF statistics -")
            for i in range(0, len(self.ids), 100000):
                try:
                    print("- ADF statistics {} -".format(i))
                    print("ids: ")
                    adf_rlt_pv = tsa.adfuller(self.ids[i : i+100000])
                    print(f'ADF statistics: {adf_rlt_pv[0]}')
                    print('p-value: {}'.format(adf_rlt_pv[1]))
                    print(f'# of lags used: {adf_rlt_pv[2]}')
                    print(f'Critical values: {adf_rlt_pv[4]}')

                    print("-"*20)
                except ValueError as e:
                    print(e)
            print("="*20)
            for i in range(0, len(self.lognum_per1s), 100000):
                try:
                    print("- ADF statistics lognum_per1s {} -".format(i))
                    print("ids: ")
                    adf_rlt_pv = tsa.adfuller(self.lognum_per1s[i : i+100000])
                    print(f'ADF statistics: {adf_rlt_pv[0]}')
                    print('p-value: {}'.format(adf_rlt_pv[1]))
                    print(f'# of lags used: {adf_rlt_pv[2]}')
                    print(f'Critical values: {adf_rlt_pv[4]}')
                    print("-" * 20)
                except ValueError as e:
                    print(e)

    def save_dict_hist(self, save_path, dict_, all_num):
        with open(save_path, mode='w') as f:
            f.writelines("\n".join(str(k) + " " + str(v)  + " " + str(v/all_num) for k, v in dict_.items()))

    def save_hist(self, save_path, results, all_num):
        print("--- Savving file=all_num: {} ---".format(all_num))
        with open(save_path, mode='w') as f:
            f.writelines("all_num: {}\n".format(all_num))
            for result in results:
                f.writelines(result[0] + " " + str(result[1]) + " " + str(100*result[1]/all_num) + "[%]\n")


    def analyze_kurtosis_hist(self, hist_right):
        hist_left = hist_right[::-1]
        histogram = hist_left + hist_right
        print("- Analysing Kurtosis of Histogram -")
        print("Kurtosis = {}".format(kurtosis(histogram)))
        print("-" * 20)

    def analyze_log_hist(self):
        print("=" * 20)
        print("=== analyze_log_hist ===")
        print("=" * 20)
        save_path = os.path.join(self.save_dir, self.log_file.split(".")[0])
        os.makedirs(save_path, exist_ok=True)
        if self.use_template:
            save_path = os.path.join(save_path, self.log_file.split(".")[0] + "_structured.txt")
        else:
            save_path = os.path.join(save_path, self.log_file.split(".")[0] + ".txt")

        results = self._count_log_types_parallel()
        all_num = sum(results.values())

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.save_hist(save_path, sorted_results, all_num)

        # 尖度の解析
        freq_list = list(results.values())
        self.analyze_kurtosis_hist(freq_list)

    def _count_log_types_parallel(self):
        log_chunks = self._split_logs_into_chunks(self.parser.df_log, chunk_size=10000)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._count_log_types, chunk) for chunk in log_chunks]
            results = Counter()
            for future in concurrent.futures.as_completed(futures):
                results.update(future.result())

        return results

    def _split_logs_into_chunks(self, df_log, chunk_size):
        chunks = []
        for i in range(0, df_log.shape[0], chunk_size):
            chunks.append(df_log.iloc[i:i + chunk_size])
        return chunks

    def _count_log_types(self, log_chunk):
        log_type_counter = Counter()
        for idx, line in log_chunk.iterrows():
            content = line['EventId'] if self.use_template else line['Content']
            log_type_counter[content] += 1
        return log_type_counter