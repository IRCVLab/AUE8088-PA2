# HYU-AUE8088, Understanding and Utilizing Deep Learning

## PA #2. Object Detection

### Important Files

```bash
├── README.md
├── requirements.txt
├── datasets
│   └── nuscenes/ (see below explanation)
├── data
│   ├── ...
│   └── nuscenes.yaml
├── models
│   ├── ...
│   ├── yolo.py
│   └── yolo5n_nuscenes.yaml
├── utils
│   ├── ...
│   ├── dataloaders.py
│   └── loss.py
├── detect.py
├── debug.ipynb
└── train_simple.py
```

### Preparation
- Prepare dataset (4.3GB, resized images with bbox labels, front camera only)
  ```bash
  $ wget https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/nuscenes_det2d.tar.gz
  $ tar xzvf nuscenes_det2d.tar.gz
  ```

- Create python virtual environment
  ```bash
  $ python3 -m venv venv/aue8088-pa2
  $ source venv/aue8088-pa2/bin/activate
  ```

- Check whether the virtual environment set properly
: The result should end with `venv/aue8088-pa2/bin/python`.

  ```bash
  $ which python
  ```

- Clone base code repository (replace `ircvlab` to `your account` if you forked the repository)
  ```bash
  $ git clone https://github.com/ircvlab/aue8088-pa2
  ```

- [!] Create a symbolic link for nuscenes dataset
    - Assume the below folder structure

      ```bash
      ├── nuscenes_det2d
      ├── aue8088-pa2
      │   ├── data/
      │   ├── models/
      │   ├── train_simple.py
      │   ├── ...
      │   └── README.md (this file)
      ```

    - Follow below commands
      ```bash
      $ cd aue8088-pa2
      $ mkdir datasets
      $ ln -s $(realpath ../nuscenes_det2d) datasets/nuscenes
      $
      ```

- Install required packages
  ```bash
  $ pip install -r requirements.txt
  ```

### Train
- Command
  ```bash
  $ python train_simple.py \
    --img 416 \
    --batch-size 64 \
    --epochs 40 \
    --data data/nuscenes.yaml \
    --cfg models/yolov5n_nuscenes.yaml \
    --weights yolov5n.pt \
    --workers 16 \
    --name yolov5n
  ```