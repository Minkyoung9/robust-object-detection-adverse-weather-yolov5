# Robust-Dbject-Detection-adverse-weather-Yolov5


## Robustness object detection in adverse weather using yolov5

<div align="center">
  <img width="100%" src="https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/1acc5fdb0f87f2b6cb978e7d297408d93199322d/resource/rain_storm-203.jpg">
</div>

### Dataset
 - BDD100k(Train, valid)
 - DAWN(Test)

### Model
Using 'Yolov5m.pt' pre-trained weights


### Train
Example


    python train_v4.py --data bdd100k2.yaml --weights yolov5m.pt --img 640 --epochs 50 --batch-size 8 --image_weights

### Evaluation
    python yolov5/detect.py --weights yolov5/runs/train/exp4/weights/best.pt --source dataset/DAWN/val --img 640 --project yolov5/runs/test


<div align="center">
  <img width="100%" src="https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/0e0e80da5730d62bca5ca5392f405a7c6579b6b6/resource/results.png">
</div>


## Updating...
