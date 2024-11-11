import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path settings
corruption_types = [
            'clean','gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ] #benchmark
results_dir = 'yolov5/runs/test/corrupted_cityscapes'  # results saved dir
ground_truth_base_dir = 'dataset/cityscapes/leftImg8bit/val'  # GT annotations path
output_csv = '6classes_evaluation_results_ours.csv'  # dir to save results

# Excluded classes list
excluded_classes = [6,7]

# filtering classes
def filter_classes(objects, excluded_classes):
    return [obj for obj in objects if obj[0] not in excluded_classes]

def calculate_iou(yolo_box1, yolo_box2):
    class_id1, x_center1, y_center1, width1, height1 = yolo_box1
    class_id2, x_center2, y_center2, width2, height2 = yolo_box2

    # YOLO format to (x1, y1, x2, y2) format
    x1 = x_center1 - width1 / 2
    y1 = y_center1 - height1 / 2
    x2 = x_center1 + width1 / 2
    y2 = y_center1 + height1 / 2

    x1_gt = x_center2 - width2 / 2
    y1_gt = y_center2 - height2 / 2
    x2_gt = x_center2 + width2 / 2
    y2_gt = y_center2 + height2 / 2

    # calculate cross-sections for Iou calculation
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



# calculate precision recall
def calculate_precision_recall(predictions, ground_truths, excluded_classes):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box in predictions:
        pred_class_id = int(pred_box[0])  # extract class IDs
        if pred_class_id in excluded_classes:
            continue
        
        matched = False
        for gt_box in ground_truths:
            gt_class_id = int(gt_box[0])  # extract class IDs
            
            iou = calculate_iou(pred_box, gt_box)  # calculate IoU
            
            if iou >= 0.5:  # IoU threshold
                if pred_class_id == gt_class_id:
                    true_positives += 1
                matched = True
                break 
        
        if not matched:
            false_positives += 1
    
    false_negatives = len(ground_truths) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall



def read_predictions(file_path, excluded_classes):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return filter_classes([(int(line.strip().split()[0]), 
                            float(line.strip().split()[1]), 
                            float(line.strip().split()[2]), 
                            float(line.strip().split()[3]), 
                            float(line.strip().split()[4])) for line in lines], excluded_classes)


def read_ground_truth(file_path, excluded_classes):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return filter_classes([(int(line.strip().split()[0]), 
                            float(line.strip().split()[1]), 
                            float(line.strip().split()[2]), 
                            float(line.strip().split()[3]), 
                            float(line.strip().split()[4])) for line in lines], excluded_classes)


def calculate_map(predictions, ground_truths, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    average_precisions = []
    
    for iou_threshold in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = len(ground_truths)
        
        for pred_box in predictions:
            matched = False
            for gt_box in ground_truths:
                iou = calculate_iou(pred_box, gt_box)
                
                if iou >= iou_threshold:
                    true_positives += 1
                    matched = True
                    false_negatives -= 1 # if correct, decrease false_negative value
                    break 

            if not matched:
                false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        average_precisions.append(precision)
    
    mAP = np.mean(average_precisions) if average_precisions else 0
    return mAP


def evaluate():
    all_precisions = {corruption: [] for corruption in corruption_types}
    all_recalls = {corruption: [] for corruption in corruption_types}
    all_map50 = {corruption: [] for corruption in corruption_types}
    all_map5095 = {corruption: [] for corruption in corruption_types}

    cities = ['frankfurt','lindau','munster']

    for corruption in corruption_types:
        print(f'Starting evaluation for {corruption}')
            
        with open(output_csv, 'a') as f:
            f.write('corruption    severity  avg_P    avg_R    avg_mAP50    avg_mAP50-95\n')

        if corruption == 'clean':
            corruption = 'gaussian_noise'
            severities = range(0,1)
        else: 
            severities = range(1,6)
        
        for severity in  severities:
            severity = str(severity)
            print(f'Severity: {severity}')

            for city in cities:
                results_sub_dir = os.path.join(results_dir,corruption,severity,city,'labels')
                ground_truth_dir = os.path.join(ground_truth_base_dir,city)
            
                for img_file in glob.glob(os.path.join(results_sub_dir, '*.txt')):                   
                    predictions = read_predictions(img_file, excluded_classes)
                    ground_truth_file = os.path.join(ground_truth_dir, os.path.basename(img_file))
                    
                    if os.path.exists(ground_truth_file):
                        ground_truths = read_ground_truth(ground_truth_file, excluded_classes)
                        precision, recall = calculate_precision_recall(predictions, ground_truths, excluded_classes)
                        all_precisions[corruption].append(precision)
                        all_recalls[corruption].append(recall)
                        
                        map50 = calculate_map(predictions, ground_truths, iou_thresholds=[0.5])
                        map5095 = calculate_map(predictions, ground_truths, iou_thresholds=np.linspace(0.5, 0.95, 10))
                        all_map50[corruption].append(map50)
                        all_map5095[corruption].append(map5095)

            # Calculate average per corruption type and severity level
            avg_precisions = {corruption: np.mean(precisions) if precisions else 0 for corruption, precisions in all_precisions.items()}
            avg_recalls = {corruption: np.mean(recalls) if recalls else 0 for corruption, recalls in all_recalls.items()}
            avg_map50 = {corruption: np.mean(map50) if map50 else 0 for corruption, map50 in all_map50.items()}
            avg_map5095 = {corruption: np.mean(map5095) if map5095 else 0 for corruption, map5095 in all_map5095.items()}
            
            with open(output_csv, 'a') as f:
                f.write(f'{corruption}  {severity}      {avg_precisions[corruption]:.3f}       {avg_recalls[corruption]:.3f}        {avg_map50[corruption]:.3f}    {avg_map5095[corruption]:.3f}\n')
        
            if severity == '0':
                break

        # Save results per severity and corruption type to CSV
        with open(output_csv, 'a') as f:
            f.write('corruption       avg_P    avg_R    avg_mAP50    avg_mAP50-95\n')
            f.write(f'{corruption}   {avg_precisions[corruption]:.3f}    {avg_recalls[corruption]:.3f}       {avg_map50[corruption]:.3f}       {avg_map5095[corruption]:.3f}\n')
            f.write('------------------------------------------------------------------\n')
        print('Done! Saved mAP results.')
        print('Calculating mAP...')

        # Print mAP results
        print(f'Average Precision: {avg_precisions[corruption]:.3f}')
        print(f'Average Recall: {avg_recalls[corruption]:.3f}')
        print(f'mAP50: {avg_map50[corruption]:.3f}')
        print(f'mAP50-95: {avg_map5095[corruption]:.3f}')


    print('Calculating mPC...')

    # mPC (mean performance across all corruption types)
    mPC_precision = np.mean(list(avg_precisions.values()))
    mPC_recall = np.mean(list(avg_recalls.values()))
    mPC_map50 = np.mean(list(avg_map50.values()))
    mPC_map5095 = np.mean(list(avg_map5095.values()))

        # Save mPC results
    with open(output_csv, 'a') as f:
        f.write(f'\n mPC Overall     {mPC_precision:.3f}    {mPC_recall:.3f}     {mPC_map50:.3f}    {mPC_map5095:.3f}\n')
    
    # Print mPC results
    print(f'mPC Average Precision: {mPC_precision:.3f}')
    print(f'mPC Average Recall: {mPC_recall:.3f}')
    print(f'mPC mAP50: {mPC_map50:.3f}')
    print(f'mPC mAP50-95: {mPC_map5095:.3f}')

    print('Saved mPC results')

if __name__ == "__main__":
    evaluate()
