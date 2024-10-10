# Robustness object detection in adverse weather using yolov5

<div align="center">
  <img width="100%" src="https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/1acc5fdb0f87f2b6cb978e7d297408d93199322d/resource/rain_storm-203.jpg">
</div>

### Dataset
 - [BDD100k](https://doc.bdd100k.com/download.html)(Train, valid)
 - [DAWN](https://data.mendeley.com/datasets/766ygrbt8y/3)(Test)

### Model
Using '[Yolov5m.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)' pre-trained weights



## How To Run ✨
install
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### Train
Example


    python train_v4.py --data bdd100k2.yaml --weights yolov5m.pt --img 640 --epochs 50 --batch-size 8 --image_weights

### Evaluation
Example


    python yolov5/detect.py --weights yolov5/runs/train/exp4/weights/best.pt --source dataset/DAWN/val --img 640 --project yolov5/runs/test


<div align="center">
  <img width="100%" src="https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/0e0e80da5730d62bca5ca5392f405a7c6579b6b6/resource/results.png">
</div>


## Updating...
