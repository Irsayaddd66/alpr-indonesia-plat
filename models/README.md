# Model Requirements

## Letakkan 2 model YOLO di folder ini:

### 1. plate_detector.pt
- **Fungsi**: Deteksi lokasi plat nomor
- **Input**: Gambar mobil/motor
- **Output**: Bounding box plat nomor
- **Classes**: 1 kelas (plate)

### 2. char_detector.pt
- **Fungsi**: Deteksi karakter di dalam plat
- **Input**: Crop plat nomor
- **Output**: Bounding box + class karakter
- **Classes**: 36 kelas
  - 0-25: A-Z
  - 26-35: 0-9

## Cara Training Model

### Stage 1: Plate Detector
```bash
yolo detect train data=plate_data.yaml model=yolov8n.pt epochs=100
```

### Stage 2: Character Detector
```bash
yolo detect train data=char_data.yaml model=yolov8n.pt epochs=100
```

## Format Dataset

### Plate Detector (plate_data.yaml)
```yaml
train: dataset/plate/images/train
val: dataset/plate/images/val
nc: 1
names: ['plate']
```

### Character Detector (char_data.yaml)
```yaml
train: dataset/char/images/train
val: dataset/char/images/val
nc: 36
names: ['A','B','C',...,'Z','0','1',...,'9']
```

## Setelah Training

Rename model hasil training:
```
runs/detect/train/weights/best.pt → models/plate_detector.pt
runs/detect/train2/weights/best.pt → models/char_detector.pt
```
