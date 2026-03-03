import os
import json
import cv2
from glob import glob

DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

def dota2coco(txt_dir, img_dir, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    print("-" * 50)
    print(f"开始诊断路径...")
    print(f"正在寻找标签文件夹: {txt_dir} \n(是否存在: {os.path.exists(txt_dir)})")
    
    txt_files = glob(os.path.join(txt_dir, '*.txt'))
    print(f"-> 找到了 {len(txt_files)} 个 .txt 标签文件！")
    
    if len(txt_files) == 0:
        print("⚠️ 警告：没找到标签文件！请检查文件夹层级。")
        return
        
    dataset = {'images': [], 'annotations': [], 'categories': []}
    for i, cls in enumerate(DOTA_CLASSES):
        dataset['categories'].append({'id': i + 1, 'name': cls})
        
    ann_id = 1
    for img_id, txt_file in enumerate(txt_files):
        filename = os.path.basename(txt_file).replace('.txt', '.png')
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(os.path.dirname(img_dir), filename)
            if not os.path.exists(img_path):
                continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        dataset['images'].append({'file_name': filename, 'id': img_id, 'width': w, 'height': h})
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) < 9: continue
                pts = [float(p) for p in parts[:8]]
                cls_name = parts[8]
                if cls_name not in DOTA_CLASSES: continue
                
                xs = pts[0::2]
                ys = pts[1::2]
                xmin, ymin = min(xs), min(ys)
                xmax, ymax = max(xs), max(ys)
                bbox_w, bbox_h = xmax - xmin, ymax - ymin
                
                cls_id = DOTA_CLASSES.index(cls_name) + 1
                
                dataset['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cls_id,
                    'segmentation': [pts], 
                    'bbox': [xmin, ymin, bbox_w, bbox_h],
                    'iscrowd': 0,
                    'area': bbox_w * bbox_h
                })
                ann_id += 1
                
    with open(json_path, 'w') as f:
        json.dump(dataset, f)
    print(f"🎉 转换成功！JSON 已保存至: {json_path}")

if __name__ == '__main__':
    # 已经修改为你剪切后的绝对路径
    base_dir = r'F:\0\Signature Work\Code\SC-main\data_use\DOTA'
    
    dota2coco(
        txt_dir=os.path.join(base_dir, r'train_split\labelTxt'),
        img_dir=os.path.join(base_dir, r'train_split\images'),
        json_path=os.path.join(base_dir, r'train_split\train_coco.json')
    )
    
    dota2coco(
        txt_dir=os.path.join(base_dir, r'val_split\labelTxt'),
        img_dir=os.path.join(base_dir, r'val_split\images'),
        json_path=os.path.join(base_dir, r'val_split\val_coco.json')
    )