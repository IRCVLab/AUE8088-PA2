import os
from collections import defaultdict

# 파일 경로 읽기
file_path = 'datasets/kaist-rgbt/train-all-04.txt'
with open(file_path, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

# 그룹화
grouped_data = defaultdict(list)
for line in lines:
    group_key = line.split('/')[-1].split('_')[0][3:5]
    grouped_data[group_key].append(line)

# 그룹 키 목록
group_keys = list(grouped_data.keys())

# k-fold 분할 및 저장
for val_key in group_keys:
    train_data = []
    val_data = grouped_data[val_key]
    
    for key in group_keys:
        if key != val_key:
            train_data.extend(grouped_data[key])

    # 파일로 저장
    train_file_path = f'datasets/kaist-rgbt/train_except_{val_key}.txt'
    val_file_path = f'datasets/kaist-rgbt/val_{val_key}.txt'

    with open(train_file_path, 'w') as train_file:
        for item in train_data:
            train_file.write(f"{item}\n")

    with open(val_file_path, 'w') as val_file:
        for item in val_data:
            val_file.write(f"{item}\n")

    print(f"Created {train_file_path} and {val_file_path}")

print("Data has been split into train and val files for each group.")
