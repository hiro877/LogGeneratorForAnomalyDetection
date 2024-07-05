#!/bin/bash

# ファイル名を引数として受け取る
input_file="$1"

# 出力ファイル名を作成（拡張子を".freq"に変更）
output_file="${input_file%.*}.freq"

# 出力ファイルを空にする
> "$output_file"

# 各行の最後の数値を取得して出力ファイルに書き込む
while IFS= read -r line; do
  # 最後の数値部分を抽出し、"[%]"を除去
  value=$(echo "$line" | awk '{print $NF}' | sed 's/\[%\]//')
  echo "$value" >> "$output_file"
done < "$input_file"
