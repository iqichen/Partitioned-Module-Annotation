import csv
import os
import cv2
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm


class AdjustedStage1Masker:

    def __init__(self, model):
        self.model = model
        self.hook_handles = []
        # 定位到更浅层的卷积（降低影响）
        self.target_layers = [
            model.backbone.conv_res_block1[0].conv,
            model.backbone.conv_res_block1[1].conv1.conv
        ]

    def _soft_mask_hook(self, module, input, output):
        # 保留60%通道特征
        mask = torch.ones_like(output)
        mask[:, ::3, :, :] = 0  # 每3个通道保留2个

        # 弱噪声干扰
        noise = torch.randn_like(output) * output.std() * 0.50

        return output * mask * 0.6 + noise * 0.4

    def enable_mask(self):
        if not self.hook_handles:
            for layer in self.target_layers:
                handle = layer.register_forward_hook(self._soft_mask_hook)
                self.hook_handles.append(handle)

    def disable_mask(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def generate_comparison(img, orig_result, ablated_result, save_path):
    """生成对比图"""

    def draw_detections(vis_img, result, color):
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        valid_dets = 0
        for score, bbox, label in zip(scores, bboxes, labels):
            if label == 0 and score > 0.3:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                valid_dets += 1
        return vis_img, valid_dets

    # 原始结果（绿色）
    orig_vis = img.copy()
    orig_vis, orig_count = draw_detections(orig_vis, orig_result, (0, 255, 0))

    # 消融结果（红色）
    ablate_vis = img.copy()
    ablate_vis, ablate_count = draw_detections(ablate_vis, ablated_result, (0, 0, 255))

    # 横向拼接
    comparison = np.concatenate((orig_vis, ablate_vis), axis=1)
    cv2.imwrite(save_path, comparison)


def main():
    # 配置路径
    config_path = r'E:\Code\Partitioned Module Annotation\mmdetection\configs\yolo\yolov3_d53_8xb8-320-273e_coco.py'
    checkpoint_path = r'E:\Code\Partitioned Module Annotation\mmdetection\checkpoints\yolov3_d53_320_273e_coco-421362b6.pth'
    img_dir = r'E:\Code\Partitioned Module Annotation\data\coco\person'
    output_dir = r'E:\Code\Partitioned Module Annotation\experiment\ablation_results'

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = init_detector(config_path, checkpoint_path, device=device)
    masker = AdjustedStage1Masker(model)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    csv_path = os.path.join(output_dir, 'ablation_report.csv')

    # 初始化统计指标
    total_orig = 0
    total_ablate = 0
    conf_diffs = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Original', 'Ablated', 'CountDiff', 'ConfidenceDiff'])

    with tqdm(total=len(img_files), desc='Processing') as pbar:
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            try:
                # 原始推理
                orig_img = mmcv.imread(img_path)
                orig_result = inference_detector(model, orig_img)

                # 消融推理
                masker.enable_mask()
                ablated_result = inference_detector(model, orig_img.copy())
                masker.disable_mask()

                # 处理结果
                def process_result(result):
                    scores = result.pred_instances.scores.cpu().numpy()
                    labels = result.pred_instances.labels.cpu().numpy()
                    valid = (labels == 0) & (scores > 0.3)
                    count = valid.sum()
                    conf = scores[valid].mean() if count > 0 else 0
                    return count, conf

                orig_count, orig_conf = process_result(orig_result)
                ablate_count, ablate_conf = process_result(ablated_result)

                # 记录数据
                total_orig += orig_count
                total_ablate += ablate_count
                if orig_count + ablate_count > 0:
                    conf_diffs.append(orig_conf - ablate_conf)

                # 保存到CSV
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        img_file,
                        orig_count,
                        ablate_count,
                        orig_count - ablate_count,
                        f"{(orig_conf - ablate_conf):.4f}" if orig_count + ablate_count > 0 else "N/A"
                    ])

                # 生成对比图
                generate_comparison(
                    orig_img,
                    orig_result,
                    ablated_result,
                    os.path.join(output_dir, f'compare_{img_file}')
                )

            except Exception as e:
                pbar.write(f"Error: {img_file} - {str(e)}")
            finally:
                pbar.update(1)

    # 终端输出汇总结果
    print("\n实验汇总结果:")
    print("═" * 40)
    print(f"总检测数 | 原始: {total_orig}  消融: {total_ablate}")
    print(f"平均置信度差异: {np.mean(conf_diffs):.4f}" if conf_diffs else "无有效置信度数据")
    print("═" * 40)
    print(f"详细报告: {csv_path}")
    print(f"对比图目录: {output_dir}")


if __name__ == "__main__":
    main()