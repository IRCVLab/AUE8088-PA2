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

# 검증 세트 주석 생성
annotations = []
for line in val_lines:
    data = line.strip().split()
    if len(data) < 6:
        print(f"Skipping line due to unexpected format: {line}")
        continue
    
    image_id = data[0]
    bbox = list(map(int, data[1:5]))  # 바운딩 박스 좌표
    label = data[5]  # 레이블
    
    annotation = {
        "image_id": image_id,
        "bbox": bbox,
        "label": label
    }
    annotations.append(annotation)

# 주석을 JSON 파일로 저장
annotation_file_path = os.path.join(data_dir, 'KAIST_annotation.json')
with open(annotation_file_path, 'w') as json_file:
    json.dump(annotations, json_file, indent=4)

print(f"훈련 세트와 검증 세트 파일이 저장되었습니다.")
print(f"검증 세트 주석 파일이 {annotation_file_path}에 저장되었습니다.")
