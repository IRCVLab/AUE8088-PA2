import os
import json

# 현재 스크립트의 디렉토리 경로를 가져옴
current_dir = os.path.dirname(__file__)

# 파일 경로 지정
train_file_path = os.path.join(current_dir, 'train.txt')
val_file_path = os.path.join(current_dir, 'val.txt')
annotation_file_path = os.path.join(current_dir, 'KAIST_annotation.json')

# train.txt 내용 출력
# with open(train_file_path, 'r') as train_file:
#     train_content = train_file.read()
#     print("train.txt 내용:")
#     print(train_content)

# val.txt 내용 출력
with open(val_file_path, 'r') as val_file:
    val_content = val_file.read()
    print("val.txt 내용:")
    print(val_content)

# KAIST_annotation.json 내용 출력
with open(annotation_file_path, 'r') as json_file:
    annotation_content = json.load(json_file)
    print("KAIST_annotation.json 내용:")
    print(json.dumps(annotation_content, indent=4))

