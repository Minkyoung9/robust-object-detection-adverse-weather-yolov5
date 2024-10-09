import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 경로 설정
results_dir = 'yolov5/runs/test'  # 결과를 저장할 디렉토리
ground_truth_base_dir = 'dataset/DAWN/val'  # 실제 주석 파일이 저장된 경로
output_csv = 'evaluation_results.csv'  # 결과를 저장할 CSV 파일

# 정밀도 및 재현율 계산 함수
def calculate_precision_recall(predictions, ground_truths):
    true_positives = len(set(predictions) & set(ground_truths))
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

# 테스트 결과 읽기
def read_predictions(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split()[0] for line in lines]  # 클래스 ID만 반환

# Ground Truth 읽기
def read_ground_truth(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split()[0] for line in lines]  # 클래스 ID만 반환

# 메인 평가 루틴
def evaluate():
    all_precisions = {k: [] for k in range(8)} # 각 클래스별로 정밀도를 저장할 리스트
    all_recalls = {k: [] for k in range(8)}    # 각 클래스별로 재현율을 저장할 리스트
    labels_mapping = {0: 'pedestrian', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle',
                        6: 'traffic light', 7: 'traffic sign'}
 
    results_sub_dir = os.path.join(results_dir, 'labels')

    for img_file in glob.glob(os.path.join(results_sub_dir, '*.txt')):
        img_name = os.path.basename(img_file).replace('.txt', '')
        
        # 예측 및 실제 값 가져오기
        predictions = read_predictions(img_file)
        ground_truth_file = os.path.join(ground_truth_base_dir, img_name + '.txt')
        
        if os.path.exists(ground_truth_file):
            ground_truths = read_ground_truth(ground_truth_file)

            # 각 클래스에 대해 정밀도 및 재현율 계산
            for label_id in range(8):  # 0부터 7까지의 클래스
                pred_for_label = [p for p in predictions if int(p) == label_id]
                gt_for_label = [g for g in ground_truths if int(g) == label_id]
                
                precision, recall = calculate_precision_recall(pred_for_label, gt_for_label)
                all_precisions[label_id].append(precision)
                all_recalls[label_id].append(recall)
    
    # 평균 정밀도와 재현율 계산
    avg_precisions = {labels_mapping[label_id]: np.mean(precisions) if precisions else 0 for label_id, precisions in all_precisions.items()}
    avg_recalls = {labels_mapping[label_id]: np.mean(recalls) if recalls else 0 for label_id, recalls in all_recalls.items()}
    
    # 결과 출력
    print(f'Average Precision: {avg_precisions}')
    print(f'Average Recall: {avg_recalls}')
    
    # 결과를 CSV 파일로 저장
    df = pd.DataFrame({
        'Class': list(avg_precisions.keys()),
        'Average Precision': list(avg_precisions.values()),
        'Average Recall': list(avg_recalls.values())
    })
    df.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

    # 그래프 생성
    plt.figure(figsize=(10, 5))
    plt.plot(list(avg_precisions.keys()), list(avg_precisions.values()), label='Average Precision', marker='o')
    plt.plot(list(avg_recalls.keys()), list(avg_recalls.values()), label='Average Recall', marker='o')
    
    plt.title('Average Precision and Recall for Each Object')
    plt.xlabel('Objects')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('precision_recall_graph.png')  # 그래프를 이미지 파일로 저장
    plt.show()

if __name__ == '__main__':
    evaluate()
