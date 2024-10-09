import os
import json

def save_as_yolo_format(destination_folder, labels, img_width, img_height, img_name):
    yolov5_format_list = []
    
    # 카테고리 매핑 (문자열 키와 숫자 값)
    category_mapping = {
        'pedestrian': 0,
        'car': 1,
        'truck': 2,
        'bus': 3,
        'motorcycle': 4,
        'bicycle': 5,
        'traffic light': 6,
        'traffic sign': 7,
    }


    for label in labels:
        category = label['category']
        
        if category in category_mapping:
            region_num = category_mapping[category]
        else:
            print(f"Warning: Category {category} not in mapping.")
            continue  # 매핑되지 않은 카테고리는 무시

        # 바운딩 박스 좌표
        x1 = label['box2d']['x1']
        y1 = label['box2d']['y1']
        x2 = label['box2d']['x2']
        y2 = label['box2d']['y2']

        # YOLOv5 포맷으로 변환
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolov5_format = [region_num, x_center, y_center, width, height]
        yolov5_format_list.append(yolov5_format)

    # 결과를 .txt 파일로 저장
    file_name = os.path.splitext(img_name)[0] + '.txt'
    file_path = os.path.join(destination_folder, file_name)

    # 해당 경로가 존재하지 않을 경우 생성
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except FileExistsError:
        print(f"Warning: Directory {os.path.dirname(file_path)} already exists.")

    with open(file_path, 'w') as file:
        for item in yolov5_format_list:
            file.write(" ".join(map(str, item)) + "\n")

def process_json(json_file_path, img_folder_path, destination_folder):
    # JSON 파일 로드
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    # 각 이미지에 대해 YOLOv5 포맷으로 변환
    for img_data in data:
        img_name = img_data['name']
        img_path = os.path.join(img_folder_path, img_name)
        if os.path.exists(img_path):
            img_width = img_data.get('width', 1280)  # 이미지 폭 가져오기
            img_height = img_data.get('height', 720)  # 이미지 높이 가져오기

            # 해당 이미지의 레이블 리스트 가져오기
            labels = img_data.get('labels', [])
            if labels:  # 레이블이 있는 경우에만 변환
                save_as_yolo_format(destination_folder, labels, img_width, img_height, img_name)
            else:
                print(f"No labels to save for {img_name}.")
        else:
            print(f"Warning: Image {img_name} not found in {img_folder_path}")

# 메인 함수
if __name__ == "__main__":
    base_path = os.path.join(os.getcwd(), "minkyoung", "dataset", "BDD100k")
    
    # 경로 설정
    train_json_path = os.path.join(base_path, "labels", "det_train.json")
    val_json_path = os.path.join(base_path, "labels", "det_val.json")
    
    train_img_path = os.path.join(base_path, "train")
    val_img_path = os.path.join(base_path, "val")
    
    train_output_path = os.path.join(base_path, "train_labels")
    val_output_path = os.path.join(base_path, "val_labels")
    
    # 출력 폴더가 없으면 생성
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)

    # JSON 파일 처리
    process_json(train_json_path, train_img_path, train_output_path)
    process_json(val_json_path, val_img_path, val_output_path)
