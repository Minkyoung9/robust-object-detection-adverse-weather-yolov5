import os
import glob
import numpy as np
import pandas as pd

# 경로 설정
results_dir = 'yolov5/runs/test/exp5'  # 결과를 저장할 디렉토리
ground_truth_base_dir = 'dataset/S-DGOD/daytime_clear'  # 실제 주석 파일이 저장된 경로
output_csv = 'sdgod_eval_results.csv'  # 결과를 저장할 CSV 파일

# 제외할 클래스 목록 설정
excluded_classes = [6, 7]

# 클래스 목록 (평가할 클래스 ID 목록)
class_names = {0: 'pedestrian', 1: 'car', 2:'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle'}


# 클래스를 필터링하는 함수 정의 (제외할 클래스를 걸러냄)
def filter_classes(objects, excluded_classes):
    return [obj for obj in objects if obj[0] not in excluded_classes]

def calculate_iou(yolo_box1, yolo_box2):
    class_id1, x_center1, y_center1, width1, height1 = yolo_box1
    class_id2, x_center2, y_center2, width2, height2 = yolo_box2

    # YOLO 포맷을 (x1, y1, x2, y2) 형식으로 변환
    x1 = x_center1 - width1 / 2
    y1 = y_center1 - height1 / 2
    x2 = x_center1 + width1 / 2
    y2 = y_center1 + height1 / 2

    x1_gt = x_center2 - width2 / 2
    y1_gt = y_center2 - height2 / 2
    x2_gt = x_center2 + width2 / 2
    y2_gt = y_center2 + height2 / 2

    # IoU 계산을 위해 교차 영역 계산
    xi1 = np.max((x1, x1_gt))
    yi1 = np.max((y1, y1_gt))
    xi2 = np.min((x2, x2_gt))
    yi2 = np.min((y2, y2_gt))

    inter_area = np.max((0, xi2 - xi1)) * np.max((0, yi2 - yi1))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou



# 정밀도 및 재현율 계산 함수 (IoU 고려)
def calculate_precision_recall(predictions, ground_truths, excluded_classes):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in predictions:
        pred_class_id = int(pred_box[0])  # 클래스 ID 추출
        if pred_class_id in excluded_classes:
            continue
        
        matched = False
        for gt_box in ground_truths:
            gt_class_id = int(gt_box[0])  # 클래스 ID 추출
            
            iou = calculate_iou(pred_box, gt_box)  # IoU 계산
            print(f"Pred Box: {pred_box}, GT Box: {gt_box}, IoU: {iou}")
            
            if iou >= 0.3:  # IoU 임계값
                if pred_class_id == gt_class_id:
                    true_positives += 1
                matched = True
                break  # 매칭된 경우 루프 종료
        
        if not matched:
            false_positives += 1
    
    false_negatives = len(ground_truths) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


# 테스트 결과 및 Ground Truth 읽기 시 제외할 클래스 필터링 적용
def read_predictions(file_path, excluded_classes):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return filter_classes([(int(line.strip().split()[0]), 
                            float(line.strip().split()[1]), 
                            float(line.strip().split()[2]), 
                            float(line.strip().split()[3]), 
                            float(line.strip().split()[4])) for line in lines], excluded_classes)

# Ground Truth 파일을 읽고 클래스 ID와 bounding box 정보를 반환
def read_ground_truth(file_path, excluded_classes):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return filter_classes([(int(line.strip().split()[0]), 
                            float(line.strip().split()[1]), 
                            float(line.strip().split()[2]), 
                            float(line.strip().split()[3]), 
                            float(line.strip().split()[4])) for line in lines], excluded_classes)

# mAP50 및 mAP50-95 계산 함수
def calculate_map(predictions, ground_truths, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    average_precisions = []
    
    for iou_threshold in iou_thresholds:
        true_positives = 0
        false_positives = 0
        
        for pred_box in predictions:
            matched = False
            for gt_box in ground_truths:
                iou = calculate_iou(pred_box, gt_box)
                
                if iou >= iou_threshold:
                    true_positives += 1
                    matched = True
                    break  # 매칭된 경우 루프 종료

            if not matched:
                false_positives += 1
        
        false_negatives = len(ground_truths) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        average_precisions.append(precision)
    
    mAP = np.mean(average_precisions) if average_precisions else 0
    return mAP


# 수정된 평가 함수에 mAP50, mAP50-95 추가
def evaluate():
    all_precisions = {class_id: [] for class_id in class_names}
    all_recalls = {class_id: [] for class_id in class_names}
    all_map50 = {class_id: [] for class_id in class_names}
    all_map5095 = {class_id: [] for class_id in class_names}

    for class_id, class_name in class_names.items():
        results_sub_dir = os.path.join(results_dir, 'labels')
        ground_truth_dir = os.path.join(ground_truth_base_dir, 'VOC2007/YOLO_annotations')

        for img_file in glob.glob(os.path.join(results_sub_dir, '*.txt')):
            img_name = os.path.basename(img_file).replace('.txt', '')

            predictions = read_predictions(img_file, excluded_classes)
            ground_truth_file = os.path.join(ground_truth_dir, img_name + '.txt')
            print(f"Ground Truth Path: {ground_truth_file}")

            if os.path.exists(ground_truth_file):
                ground_truths = read_ground_truth(ground_truth_file, excluded_classes)

                # 현재 클래스에 해당하는 예측 및 Ground Truth 필터링
                filtered_predictions = [pred for pred in predictions if int(pred[0]) == class_id]
                filtered_ground_truths = [gt for gt in ground_truths if int(gt[0]) == class_id]

                # 정밀도 및 재현율 계산
                precision, recall = calculate_precision_recall(filtered_predictions, filtered_ground_truths, excluded_classes)
                all_precisions[class_id].append(precision)
                all_recalls[class_id].append(recall)

                # mAP50 및 mAP50-95 계산
                map50 = calculate_map(filtered_predictions, filtered_ground_truths, iou_thresholds=[0.5])
                map5095 = calculate_map(filtered_predictions, filtered_ground_truths, iou_thresholds=np.linspace(0.5, 0.95, 10))
                all_map50[class_id].append(map50)
                all_map5095[class_id].append(map5095)

    # 평균 계산
    avg_precisions = {class_id: np.mean(precisions) if precisions else 0 for class_id, precisions in all_precisions.items()}
    avg_recalls = {class_id: np.mean(recalls) if recalls else 0 for class_id, recalls in all_recalls.items()}
    avg_map50 = {class_id: np.mean(maps) if maps else 0 for class_id, maps in all_map50.items()}
    avg_map5095 = {class_id: np.mean(maps) if maps else 0 for class_id, maps in all_map5095.items()}

    # 결과 출력
    results = {'Class': [], 'Precision': [], 'Recall': [], 'mAP50': [], 'mAP50-95': []}
    for class_id, class_name in class_names.items():
        results['Class'].append(class_name)
        results['Precision'].append(avg_precisions[class_id])
        results['Recall'].append(avg_recalls[class_id])
        results['mAP50'].append(avg_map50[class_id])
        results['mAP50-95'].append(avg_map5095[class_id])

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(results_df)

evaluate()
