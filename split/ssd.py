import os
# 设置环境变量，避免KMeans在Windows上的内存泄漏警告
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import mmcv
import numpy as np
import pickle
from mmdet.apis import init_detector
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import re

# 参数解析
parser = argparse.ArgumentParser(description='使用K-means聚类拆分SSD模型')
parser.add_argument('--k', type=int, default=3, help='分区数量K')
parser.add_argument('--n_init', type=int, default=10, help='K-means运行次数')
parser.add_argument('--max_iter', type=int, default=300, help='最大迭代次数')
args = parser.parse_args()

# 硬编码路径配置
SAVE_DIR = r"E:\Code\Partitioned Module Annotation\features"
os.makedirs(SAVE_DIR, exist_ok=True)
config_file = r'E:\Code\Partitioned Module Annotation\mmdetection\configs\ssd\ssd300_coco.py'
checkpoint_file = r'E:\Code\Partitioned Module Annotation\mmdetection\checkpoints\ssd300_coco_20210803_015428-d231a06e.pth'
img_folder = r'E:\Code\Partitioned Module Annotation\data\coco\laptop'

class BatchFeatureHook:
    def __init__(self):
        self.batch_features = []
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        feat = output.detach().cpu().numpy()
        self.batch_features.append(feat)

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 定义要提取特征的层
def get_all_layers(model, prefix=''):
    """递归获取模型中的所有层"""
    layers = {}
    for name, module in model.named_children():
        current_prefix = f"{prefix}.{name}" if prefix else name
        
        # 添加当前模块
        layers[current_prefix] = module
        
        # 递归处理子模块
        sub_layers = get_all_layers(module, current_prefix)
        layers.update(sub_layers)
    
    return layers

# 获取模型中的所有层
all_layers = get_all_layers(model.backbone)
print(f"找到 {len(all_layers)} 个层")

# 选择特定层进行特征提取
def select_feature_layers(layers):
    """选择SSD中的特征层"""
    selected_layers = {}
    feature_pattern = re.compile(r'features\.(\d+)')
    
    for layer_name, layer in layers.items():
        match = feature_pattern.search(layer_name)
        if match and isinstance(layer, torch.nn.Conv2d):
            layer_idx = int(match.group(1))
            # SSD中的关键特征层通常是卷积层
            if layer_idx in [16, 24, 26, 28, 30, 32]:  # 这些是SSD300中的关键层索引
                selected_layers[f"layer_{layer_idx}"] = layer
    
    return selected_layers

# 选择特征层
selected_layers = select_feature_layers(all_layers)
print(f"选择了 {len(selected_layers)} 个特征层进行特征提取")

# 注册钩子
hooks = {}
for layer_name, layer in selected_layers.items():
    hook = BatchFeatureHook()
    handle = layer.register_forward_hook(hook.hook_fn)
    hooks[layer_name] = (hook, handle)

def prepare_image(img):
    img = mmcv.imresize(img, (300, 300))
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda().float()

# 获取图片
img_paths = [os.path.join(img_folder, f) for f in sorted(os.listdir(img_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]

# 预处理
original_imgs = []
processed_imgs = []
for path in tqdm(img_paths, desc='预处理'):
    img = mmcv.imread(path)
    original_imgs.append(img)
    processed_imgs.append(prepare_image(img))

# 特征提取
with torch.no_grad():
    for img in tqdm(processed_imgs, desc='提取特征'):
        _ = model.backbone(img)

# 收集特征数据
module_features = {mod: hook.batch_features[:10] for mod, (hook, _) in hooks.items()}

# 计算层间余弦相似度
def calculate_feature_vectors(module_features):
    """将每个模块的特征转换为适合聚类的向量"""
    feature_vectors = {}
    feature_shapes = {}
    
    # 对每个模块，计算特征向量
    for mod_name, features in module_features.items():
        if not features:
            continue
            
        # 对每个模块，我们将所有批次的特征统一处理
        all_features = []
        
        for batch_idx, batch_feat in enumerate(features):
            # 记录特征形状
            if batch_idx == 0:
                feature_shapes[mod_name] = batch_feat.shape
                
            # 将特征展平为一维向量
            # 首先确保我们处理的是4D张量 [batch, channels, height, width]
            if len(batch_feat.shape) == 4:
                # 对于每个样本，我们计算通道维度的统计特征
                # 计算每个通道的均值和标准差
                channel_means = np.mean(batch_feat, axis=(2, 3))  # [batch, channels]
                channel_stds = np.std(batch_feat, axis=(2, 3))    # [batch, channels]
                
                # 将均值和标准差拼接在一起
                channel_stats = np.concatenate([channel_means, channel_stds], axis=1)  # [batch, channels*2]
                
                # 对所有样本取平均
                feature_vector = np.mean(channel_stats, axis=0)  # [channels*2]
                all_features.append(feature_vector)
            else:
                # 对于非4D张量，我们尝试将其展平并取统计特征
                flat_feat = batch_feat.reshape(batch_feat.shape[0], -1)
                feature_vector = np.mean(flat_feat, axis=0)
                all_features.append(feature_vector)
        
        # 计算所有批次的平均向量
        if all_features:
            feature_vectors[mod_name] = np.mean(all_features, axis=0)
    
    return feature_vectors, feature_shapes

# 执行K-means聚类
def cluster_modules(feature_vectors, k=args.k, n_init=args.n_init, max_iter=args.max_iter):
    """使用K-means聚类模块"""
    # 准备数据
    module_names = list(feature_vectors.keys())
    
    # 确保所有特征向量具有相同的维度
    vector_lengths = [len(feature_vectors[name]) for name in module_names]
    
    if len(set(vector_lengths)) > 1:
        # 如果维度不一致，我们需要进行处理
        # 找到最小维度
        min_length = min(vector_lengths)
        
        # 截断所有向量到最小维度
        for name in module_names:
            feature_vectors[name] = feature_vectors[name][:min_length]
    
    # 将特征向量转换为numpy数组
    X = np.array([feature_vectors[name] for name in module_names])
    
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(X)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # 整理聚类结果
    cluster_results = {}
    for i, module_name in enumerate(module_names):
        cluster_id = int(clusters[i])
        if cluster_id not in cluster_results:
            cluster_results[cluster_id] = []
        cluster_results[cluster_id].append(module_name)
    
    # 重新排序聚类结果，确保分区编号从0开始连续
    sorted_cluster_results = {}
    for i, (cluster_id, modules) in enumerate(sorted(cluster_results.items())):
        sorted_cluster_results[i] = modules
    
    return sorted_cluster_results, similarity_matrix, module_names

# 计算特征向量
feature_vectors, feature_shapes = calculate_feature_vectors(module_features)

# 执行聚类
cluster_results, similarity_matrix, clustered_modules = cluster_modules(feature_vectors)

# 显示聚类结果
print("\n拆分结果:")
for cluster_id, modules in cluster_results.items():
    print(f"module{cluster_id+1}：{modules}")

# 保存数据
intermediate_data = {
    'module_features': module_features,
    'original_imgs': original_imgs,
    'module_names': clustered_modules,
    'cluster_results': cluster_results,
    'similarity_matrix': similarity_matrix,
    'feature_vectors': feature_vectors,
    'feature_shapes': feature_shapes
}

# 保存中间数据
output_file = os.path.join(SAVE_DIR, f'ssd_features_{args.k}.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(intermediate_data, f)

# 清理钩子
for _, handle in hooks.values():
    handle.remove()
print(f"\n特征和聚类结果保存至 {output_file}")