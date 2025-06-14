import os
import requests
import json
import time
import csv
import base64
from pathlib import Path
import re

class QwenAnnotator:
    """
    使用千问大模型对模块热力图进行功能标注
    """
    
    def __init__(self):
        # 标准功能列表
        self.standard_functions = {
            "一、边缘与基础感知": [
                "1. 边缘检测",
                "2. 纹理识别",
                "3. 颜色感知",
                "4. 光照适应"
            ],
            "二、目标检测与结构分析": [
                "1. 物体检测",
                "2. 边界框回归",
                "3. 关键点检测",
                "4. 姿态估计"
            ],
            "三、语义与高层推理": [
                "1. 语义分割",
                "2. 场景分类",
                "3. 关系建模",
                "4. 多模态融合"
            ]
        }
        
        # 输出文件夹
        self.output_dir = Path("neo4j")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def encode_image(self, image_path):
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_prompt(self, model_name, module_name, categories=None, responses=None):
        """生成发送给千问大模型的prompt"""
        prompt = f"这是{model_name}的{module_name}\n"
        
        if categories and responses:
            prompt += f"请根据这10张热力图，分别对应类别{', '.join(categories)}，通道平均响应度为{', '.join([f'{r:.4f}' for r in responses])}\n"
        else:
            prompt += "请根据这10张热力图\n"
            
        prompt += """执行以下操作：

1. 核心功能标注
为每个模块生成1个主要功能标签，格式：
"[类别]_[功能]_模块"
功能从12个标准功能里选

2. 判断依据要求
- 关键关注区域（如：人/交通工具/户外物品）
- 跨图像响应模式（如：在8/10图中对动物有强响应）

3. 输出格式
- 功能标签：[_]_[_]_模块
- 置信度：（0-1）

标准功能列表：
一、边缘与基础感知
1. 边缘检测
2. 纹理识别
3. 颜色感知
4. 光照适应
二、目标检测与结构分析
1. 物体检测
2. 边界框回归
3. 关键点检测
4. 姿态估计
三、语义与高层推理
1. 语义分割
2. 场景分类
3. 关系建模
4. 多模态融合"""
        
        return prompt
    
    def extract_annotation_result(self, response_text):
        """从千问大模型的回复中提取标注结果"""
        # 提取功能标签
        label_match = re.search(r"功能标签：\s*(\S+)", response_text)
        label = label_match.group(1) if label_match else "未知_未知_模块"
        
        # 提取置信度
        confidence_match = re.search(r"置信度：\s*(\d+\.\d+|\d+)", response_text)
        confidence = confidence_match.group(1) if confidence_match else "0"
        
        # 提取判断依据（可能跨多行）
        reasoning = ""
        reasoning_section = re.search(r"判断依据[：:]([\s\S]+?)(?=功能标签|$)", response_text)
        if reasoning_section:
            reasoning = reasoning_section.group(1).strip()
        
        return {
            "label": label,
            "confidence": float(confidence),
            "reasoning": reasoning
        }
    
    def annotate_module(self, model_name, module_path):
        """使用千问大模型标注单个模块"""
        print(f"正在标注: {module_path}")
        
        # 获取模块名称
        module_name = os.path.basename(module_path).split('.')[0]
        
        # 编码图像
        image_base64 = self.encode_image(module_path)
        
        # 生成prompt
        prompt = self.generate_prompt(model_name, module_name)
        
        # 这里应该实现与千问大模型的API交互
        # 由于无法直接调用千问API，这里提供指导说明
        print(f"请访问 https://chat.qwen.ai/ 并上传图片")
        print(f"使用以下prompt:\n{prompt}")
        
        # 在实际应用中，这里应该有API调用代码
        # response = self.call_qwen_api(image_base64, prompt)
        
        # 由于无法直接调用API，这里模拟用户手动输入结果
        print("请将千问大模型的回复粘贴到这里（输入'END'结束）：")
        lines = []
        while True:
            line = input()
            if line == 'END':
                break
            lines.append(line)
        
        response_text = '\n'.join(lines)
        
        # 提取标注结果
        result = self.extract_annotation_result(response_text)
        result["module"] = module_name
        result["model"] = model_name
        
        return result
    
    def annotate_model(self, model_name, heatmaps_dir):
        """标注一个模型的所有模块"""
        results = []
        
        # 获取该模型的所有热力图
        heatmap_path = Path(heatmaps_dir) / model_name
        if not heatmap_path.exists():
            print(f"错误: 未找到模型 {model_name} 的热力图目录")
            return results
        
        # 遍历所有热力图文件
        for heatmap_file in sorted(heatmap_path.glob("*.jpg")):
            result = self.annotate_module(model_name, str(heatmap_file))
            results.append(result)
        
        return results
    
    def save_results_to_csv(self, results, output_file="module_annotations.csv"):
        """将标注结果保存为CSV文件"""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model', 'module', 'label', 'confidence', 'reasoning']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"标注结果已保存至: {output_path}")
        return output_path

def main():
    # 创建标注器实例
    annotator = QwenAnnotator()
    
    # 指定要标注的模型
    model_name = input("请输入要标注的模型名称（例如：yolov3）: ")
    
    # 标注模型
    results = annotator.annotate_model(model_name, "heatmaps")
    
    # 保存结果
    if results:
        output_file = f"{model_name}_annotations.csv"
        annotator.save_results_to_csv(results, output_file)

if __name__ == "__main__":
    main()
