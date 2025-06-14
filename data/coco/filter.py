from pycocotools.coco import COCO
import os
import shutil

# 配置路径
dataDir = r'E:\Code\Partitioned Module Annotation'
dataType = 'train2017'
annFile = os.path.join(dataDir, 'instances_train2017.json')
image_dir = os.path.join(dataDir, dataType)  # 图片目录直接指向 train2017
output_dir = os.path.join(dataDir, 'selected_images')

# 初始化 COCO API
coco = COCO(annFile)

# 目标类别列表（必须与 COCO 标注中的名称完全一致）
target_classes = [
    'person', 'car', 'traffic light', 'bird', 'backpack',
    'sports ball', 'bottle', 'apple', 'couch', 'laptop'
]

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历每个类别
for cls in target_classes:
    # 获取类别 ID（严格匹配名称）
    catIds = coco.getCatIds(catNms=[cls])
    if not catIds:
        print(f"Error: 类别 '{cls}' 不存在于标注文件中！")
        continue
    cat_id = catIds[0]

    # 获取图片 ID
    imgIds = coco.getImgIds(catIds=[cat_id])
    if len(imgIds) == 0:
        print(f"Error: 类别 '{cls}' 无可用图片！")
        continue

    # 创建类别文件夹（英文名称，替换空格为下划线）
    cls_folder = os.path.join(output_dir, cls.replace(' ', '_'))
    os.makedirs(cls_folder, exist_ok=True)

    # 提取最多10张图片
    count = 0
    for img_id in imgIds:
        if count >= 10:
            break
        # 加载图片信息
        img = coco.loadImgs(img_id)[0]
        src_path = os.path.join(image_dir, img['file_name'])

        # 检查图片是否存在
        if not os.path.exists(src_path):
            print(f"Missing: {src_path}")
            continue

        # 复制图片
        dst_path = os.path.join(cls_folder, img['file_name'])
        shutil.copy(src_path, dst_path)
        print(f"Copied: {cls}/{img['file_name']}")
        count += 1

    print(f"类别 '{cls}' 已保存 {count} 张图片")

print("处理完成！输出目录:", output_dir)