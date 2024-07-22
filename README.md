# PCDS2024
使用したデータセットは以下から取得できます。
https://drive.google.com/file/d/1ucR8i7xQrfD5FzdQrprQT-ode2P8m4Ab/view?usp=sharing

## 実行順序
論文中の提案手法Exp1~3は次のスクリプトから実行します。
- Train
configration/exp1.jsonのonly_testをfalseに設定する
python main_param_ad.py --config configurations/exp1.json 
- Test
configration/exp1.jsonのonly_testをtrueに設定する
python main_param_ad.py --config configurations/exp1.json 

比較手法は次のスクリプトから実行します。
python compatitive_unsupervised_main.py  --model_type isolationforest --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256


# Under Construction For Future
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