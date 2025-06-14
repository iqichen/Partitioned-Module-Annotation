import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import argparse

# 参数解析
parser = argparse.ArgumentParser(description='为MS R-CNN模型生成热力图')
parser.add_argument('--k', type=int, default=3, help='热力图数量K，默认为3')
args = parser.parse_args()

# 配置路径
FEATURES_FILE = r'E:\Code\Partitioned Module Annotation\features\ms_rcnn_features.pkl'
HEATMAPS_DIR = r'E:\Code\Partitioned Module Annotation\heatmaps\ms_rcnn'
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# 加载特征数据
with open(FEATURES_FILE, 'rb') as f:
    data = pickle.load(f)


def generate_heatmap(feat_map, orig_img, upsample_ratio):
    """生成单张热力图（修改后的通道选择逻辑）"""
    # 计算每个通道的响应值（均值）和标准差
    channel_means = np.mean(feat_map, axis=(1, 2))
    channel_std = np.std(feat_map, axis=(1, 2))
    
    # 排序通道响应值
    sorted_indices = np.argsort(channel_means)
    num_channels = len(sorted_indices)
    
    # 1. 去掉响应值最低的30%和最高的10%
    low_cutoff = int(num_channels * 0.3)
    high_cutoff = int(num_channels * 0.9)
    filtered_indices = sorted_indices[low_cutoff:high_cutoff] if low_cutoff < high_cutoff else sorted_indices
    
    # 2. 归一化响应值和标准差
    filtered_means = channel_means[filtered_indices]
    filtered_std = channel_std[filtered_indices]
    
    norm_means = np.ones_like(filtered_means) * 0.5
    norm_std = np.ones_like(filtered_std) * 0.5
    
    if np.max(filtered_means) != np.min(filtered_means):
        norm_means = (filtered_means - np.min(filtered_means)) / (np.max(filtered_means) - np.min(filtered_means))
    
    if np.max(filtered_std) != np.min(filtered_std):
        norm_std = (filtered_std - np.min(filtered_std)) / (np.max(filtered_std) - np.min(filtered_std))
    
    # 3. 加权计算：响应值*0.5 + 标准差*0.5
    channel_scores = 0.5 * norm_means + 0.5 * norm_std
    
    # 4. 选择加权分数最高的top5通道
    top_indices_in_filtered = np.argsort(-channel_scores)[:min(5, len(filtered_indices))]
    top_indices = filtered_indices[top_indices_in_filtered]
    
    if len(top_indices) == 0:
        top_indices = np.argsort(-channel_means)[:min(5, num_channels)]
    
    # 生成加权特征图
    weighted_feat = np.sum(feat_map[top_indices], axis=0)
    
    # 归一化处理
    heatmap = cv2.resize(weighted_feat, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 融合原始图像
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_img, 0.6, heatmap_img, 0.4, 0)
    
    return blended, np.mean(channel_means[top_indices])


def process_module_features(module_id, module_layers, module_features, original_imgs):
    """处理一个模块的所有特征，生成一个热力图网格"""
    grid_rows = []
    all_responses = []
    
    # 为每个图片生成热力图
    for img_idx in range(min(10, len(original_imgs))):
        # 选择一个层来生成热力图
        selected_layer = None
        selected_feature = None
        
        for layer_name in module_layers:
            if layer_name in module_features:
                features = module_features[layer_name]
                if img_idx < len(features) and features[img_idx] is not None:
                    selected_layer = layer_name
                    selected_feature = features[img_idx]
                    break
        
        if selected_layer and selected_feature is not None:
            # 获取特征图和原始图像
            feat_map = selected_feature.squeeze(0)
            orig_img = original_imgs[img_idx]
            
            # 计算upsample_ratio
            ratio = 1
            if 'feature_shapes' in data and selected_layer in data['feature_shapes']:
                shape = data['feature_shapes'][selected_layer]
                if len(shape) >= 4:
                    ratio = orig_img.shape[1] / shape[3]
            
            # 生成热力图
            if feat_map.ndim == 3:
                heatmap, response = generate_heatmap(feat_map, orig_img, ratio)
                grid_rows.append(cv2.resize(heatmap, (256, 256)))
                all_responses.append(response)
            else:
                grid_rows.append(np.zeros((256, 256, 3), dtype=np.uint8))
        else:
            grid_rows.append(np.zeros((256, 256, 3), dtype=np.uint8))
    
    # 补全网格（不足10张时用黑图填充）
    while len(grid_rows) < 10:
        grid_rows.append(np.zeros((256, 256, 3), dtype=np.uint8))
    
    # 创建网格图像并保存
    grid_image = np.vstack([np.hstack(grid_rows[:5]), np.hstack(grid_rows[5:])])
    output_path = os.path.join(HEATMAPS_DIR, f"module{module_id}_laptop.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
    
    return np.mean(all_responses) if all_responses else None


# 主处理流程
if __name__ == "__main__":
    # 处理每个聚类
    all_module_responses = {}
    
    # 确保k不超过聚类数量
    k = min(args.k, len(data['cluster_results']))
    
    for cluster_id in range(k):
        # 获取该聚类的所有层名称
        module_layers = data['cluster_results'][cluster_id]
        
        if module_layers:
            # 显示处理信息
            res_keys = [data['layer_to_res_key'][name] for name in module_layers if name in data['layer_to_res_key']]
            print(f"处理模块 {cluster_id+1}，包含层: {res_keys}")
            
            # 处理模块特征并生成热力图
            module_avg = process_module_features(
                cluster_id+1,  # 模块ID从1开始
                module_layers,
                data['module_features'],
                data['original_imgs']
            )
            
            # 收集响应值
            all_module_responses[f"module{cluster_id+1}"] = module_avg
    
    # 打印结果
    print("\n各模块平均响应值：")
    for module_name, module_avg in all_module_responses.items():
        if module_avg is not None:
            print(f"{module_name}: {module_avg:.4f}")
        else:
            print(f"{module_name}: 数据不足")
    
    print(f"\n热力图已保存至: {HEATMAPS_DIR}")