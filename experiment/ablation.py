import csv
import os
import cv2
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage


class Stage1EdgeMasker:
    """专注于模拟阶段1边缘检测能力的消融器"""

    def __init__(self, model):
        self.model = model
        self.hook_handles = []
        # 特别针对边缘检测的第一阶段卷积层
        self.target_layers = [
            model.backbone.conv_res_block1[0].conv,  # 第一阶段的初始卷积
        ]

    def _edge_suppress_hook(self, module, input, output):
        """专门抑制边缘检测特征的钩子函数"""
        # 使用高斯模糊削弱边缘特征
        batch_size, channels, height, width = output.shape
        
        # 将特征转到CPU进行处理
        output_np = output.detach().cpu().numpy()
        blurred_output = np.zeros_like(output_np)
        
        # 对每个通道应用高斯模糊
        for b in range(batch_size):
            for c in range(channels):
                # 使用高斯滤波削弱边缘特征
                blurred_output[b, c] = ndimage.gaussian_filter(output_np[b, c], sigma=1.5)
        
        # 转回原设备并保持梯度
        blurred_tensor = torch.tensor(blurred_output, device=output.device)
        
        # 保留70%原始特征，30%模糊特征
        return output * 0.7 + blurred_tensor * 0.3

    def enable_mask(self):
        if not self.hook_handles:
            for layer in self.target_layers:
                handle = layer.register_forward_hook(self._edge_suppress_hook)
                self.hook_handles.append(handle)

    def disable_mask(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def generate_edge_comparison(img, orig_result, ablated_result, save_path):
    """生成包含边缘提取的对比图"""
    # 图像预处理，转为灰度并提取边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 模拟消融后的边缘检测效果（使用高斯模糊）
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    blurred_edges = cv2.Canny(blurred_gray, threshold1=50, threshold2=150)
    blurred_edges_color = cv2.cvtColor(blurred_edges, cv2.COLOR_GRAY2BGR)
    
    def draw_detections(vis_img, result, color):
        # 所有检测都视为缺陷
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        valid_dets = 0
        for score, bbox in zip(scores, bboxes):
            if score > 0.2:  # 置信度阈值设为0.2
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                # 添加置信度文本
                text = f"{score:.2f}"
                cv2.putText(vis_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                valid_dets += 1
        return vis_img, valid_dets

    # 原始结果（绿色）
    orig_vis = img.copy()
    orig_vis, orig_count = draw_detections(orig_vis, orig_result, (0, 255, 0))

    # 消融结果（红色）
    ablate_vis = img.copy()
    ablate_vis, ablate_count = draw_detections(ablate_vis, ablated_result, (0, 0, 255))

    # 添加标题
    cv2.putText(orig_vis, f"Original: {orig_count} defects", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(ablate_vis, f"Edge-Ablated: {ablate_count} defects", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(edges_color, "Original Edge Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(blurred_edges_color, "Ablated Edge Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # 创建四图拼接布局
    top_row = np.concatenate((orig_vis, ablate_vis), axis=1)
    bottom_row = np.concatenate((edges_color, blurred_edges_color), axis=1)
    comparison = np.vstack((top_row, bottom_row))
    
    cv2.imwrite(save_path, comparison)
    return orig_count, ablate_count


def main():
    # 配置路径
    config_path = r'E:\Code\Partitioned Module Annotation\mmdetection\configs\yolo\yolov3_d53_8xb8-320-273e_coco.py'
    checkpoint_path = r'E:\Code\Partitioned Module Annotation\mmdetection\checkpoints\yolov3_d53_320_273e_coco-421362b6.pth'
    img_dir = r'E:\Code\Partitioned Module Annotation\data\MVTEC\bottle'
    output_dir = r'E:\Code\Partitioned Module Annotation\experiment\ablation_results_mvtec_bottle'

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = init_detector(config_path, checkpoint_path, device=device)
    edge_masker = Stage1EdgeMasker(model)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    csv_path = os.path.join(output_dir, 'edge_ablation_report.csv')

    # 初始化统计指标
    total_orig = 0
    total_ablate = 0
    conf_diffs = []
    all_results = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Original', 'EdgeAblated', 'CountDiff', 'PercentageChange', 'ConfidenceDiff'])

    with tqdm(total=len(img_files), desc='Processing Plate Edge Defects') as pbar:
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            try:
                # 原始推理
                orig_img = mmcv.imread(img_path)
                orig_result = inference_detector(model, orig_img)

                # 边缘消融推理
                edge_masker.enable_mask()
                ablated_result = inference_detector(model, orig_img.copy())
                edge_masker.disable_mask()

                # 处理结果
                def process_result(result):
                    scores = result.pred_instances.scores.cpu().numpy()
                    valid = scores > 0.2  # 置信度阈值设为0.2
                    count = valid.sum()
                    conf = scores[valid].mean() if count > 0 else 0
                    # 即使没有检测到目标，也返回最高置信度
                    max_conf = scores.max() if len(scores) > 0 else 0
                    return count, conf, max_conf

                orig_count, orig_conf, orig_max_conf = process_result(orig_result)
                ablate_count, ablate_conf, ablate_max_conf = process_result(ablated_result)
                
                # 计算百分比变化
                percent_change = ((orig_count - ablate_count) / orig_count * 100) if orig_count > 0 else 0

                # 记录数据
                total_orig += orig_count
                total_ablate += ablate_count
                all_results.append((orig_count, ablate_count))
                
                # 无论是否检测到目标，都记录置信度差异
                if orig_count + ablate_count > 0:
                    conf_diffs.append(orig_conf - ablate_conf)
                else:
                    # 使用最大置信度进行比较
                    conf_diffs.append(orig_max_conf - ablate_max_conf)

                # 保存到CSV
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_file,
                        orig_count,
                        ablate_count,
                        orig_count - ablate_count,
                        f"{percent_change:.1f}%",
                        # 无论是否有检测到目标，都输出置信度差异
                        f"{(orig_conf - ablate_conf):.4f}" if orig_count + ablate_count > 0 else f"{(orig_max_conf - ablate_max_conf):.4f}"
                    ])

                # 生成对比图
                result_counts = generate_edge_comparison(
                    orig_img,
                    orig_result,
                    ablated_result,
                    os.path.join(output_dir, f'edge_compare_{img_file}')
                )

            except Exception as e:
                pbar.write(f"Error: {img_file} - {str(e)}")
            finally:
                pbar.update(1)

    # 计算总体影响
    detection_rate = (total_ablate / total_orig * 100) if total_orig > 0 else 0
    detection_drop = 100 - detection_rate

    # 终端输出汇总结果
    print("\n边缘检测能力消融实验汇总结果:")
    print("═" * 50)
    print(f"检测统计 | 原始: {total_orig}  边缘消融后: {total_ablate}")
    print(f"检测下降: {total_orig - total_ablate} ({detection_drop:.1f}% 降低)")
    print(f"平均置信度差异: {np.mean(conf_diffs):.4f}" if conf_diffs else "无有效置信度数据")
    print("═" * 50)
    print(f"详细报告: {csv_path}")
    print(f"对比图目录: {output_dir}")


if __name__ == "__main__":
    main() 