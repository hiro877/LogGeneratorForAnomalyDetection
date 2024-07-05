LogGeneratorForAnomalyDetection


このプロジェクトはテキストログを用いた異常検知の研究領域に不足している
データセットを自動生成するツールを作成するプロジェクトです。

研究領域で使用されている主要なデータセットは1行毎のログに対するラベルしかないため、
１行のログに対する異常検知またはシーケンス異常検知の研究しか行うことが出来ませんでした。

そこで、ソフトウェア開発現場で頻繁に発生するパラメータ異常向けのデータセットの作成も行います。

■作成するデータセットの概要
１．テンプレートごとの異常ラベル　main.py
２．パラメータの異常ラベル　
python main.py -ip -pf parameter_logs/generated_logs/dataset_test_info.txt
事前に、下記ファイルを用いてパラメータ異常用のContentsを作成
make_parameter_log_contents.py
〇任意のログと同じ出現頻度構造のログデータセットを作成
1. analyze.py
2. investigate_analyzed_file.py
３．テンプレートごとの異常ラベル make_mimic_logs.py
４．パラメータの異常ラベル　
3. python make_mimic_logs.py -fref results/Android/Android_structured.freq