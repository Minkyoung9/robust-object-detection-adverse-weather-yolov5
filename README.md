# Robust-Dbject-Detection-adverse-weather-Yolov5


## Robustness object detection in adverse weather using yolov5
[rain_storm-203.jpg](https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/1acc5fdb0f87f2b6cb978e7d297408d93199322d/resource/rain_storm-203.jpg)

### Dataset
 - BDD100k(Train, valid)
 - DAWN(Test)

### Model
Using 'Yolov5m.pt' pre-trained weights


### Train
Example
    python train_v4.py --data bdd100k2.yaml --weights yolov5m.pt --img 640 --epochs 50 --batch-size 8 --image_weights

### Val
[Result.png](https://github.com/Minkyoung9/robust-object-detection-adverse-weather-yolov5/blob/539c43841a1716af89561c06c25df431465769d5/resource/results.png)

### Evaluation
    python yolov5/detect.py --weights yolov5/runs/train/exp4/weights/best.pt --source dataset/DAWN/val --img 640 --project yolov5/runs/test


## Updating...
