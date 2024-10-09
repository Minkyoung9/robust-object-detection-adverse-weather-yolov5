import os
import xml.etree.ElementTree as ET

# 클래스 ID 매핑
classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]  # YOLO에 사용할 클래스

def convert_bbox(size, box):
    # size는 이미지의 (width, height)
    # box는 (xmin, xmax, ymin, ymax)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, yolo_output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(yolo_output_path, 'w') as out_file:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
            bbox = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bbox])}\n")

def convert_sdgod_to_yolo(sdgod_root):
    voc_dirs = ['daytime_clear', 'daytime_foggy', 'dusk_rainy', 'night_rainy', 'night_sunny']
    
    for voc_dir in voc_dirs:
        annotations_dir = os.path.join(sdgod_root, voc_dir, 'VOC2007', 'Annotations')
        images_dir = os.path.join(sdgod_root, voc_dir, 'VOC2007', 'JPEGImages')
        output_dir = os.path.join(sdgod_root, voc_dir, 'YOLOAnnotations')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for xml_file in os.listdir(annotations_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotations_dir, xml_file)
                image_id = os.path.splitext(xml_file)[0]
                yolo_output_path = os.path.join(output_dir, f"{image_id}.txt")
                convert_annotation(xml_path, yolo_output_path)
                print(f"Converted {xml_file} to YOLO format.")

# 사용 예시
sdgod_root = '/home/intern/minkyoung/dataset/S-DGOD'
convert_sdgod_to_yolo(sdgod_root)
