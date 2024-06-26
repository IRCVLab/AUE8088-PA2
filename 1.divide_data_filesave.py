import os
import json
import random

# 경로 설정
data_dir = 'datasets/kaist-rgbt'
train_file_path = os.path.join(data_dir, 'train-all-04.txt')

# train-all-04.txt 파일 읽기
with open(train_file_path, 'r') as file:
    lines = file.readlines()

# 데이터 섞기
random.shuffle(lines)

# 훈련 세트와 검증 세트 분할 (80%:20%)
split_ratio = 0.8
split_index = int(len(lines) * split_ratio)
train_lines = lines[:split_index]
val_lines = lines[split_index:]

# 분할된 데이터를 파일로 저장
with open(os.path.join(data_dir, 'train.txt'), 'w') as train_file:
    train_file.writelines(train_lines)

with open(os.path.join(data_dir, 'val.txt'), 'w') as val_file:
    val_file.writelines(val_lines)
