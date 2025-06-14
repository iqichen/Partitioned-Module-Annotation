# 大模型模块化功能标注平台

## 项目简介

本项目致力于对大型语言模型进行模块化拆分，并对各个模块赋予语义标注。通过聚类算法实现模型能力的细粒度划分，结合热力图可视化与交互式标注，最终将标注结果导入Neo4j，构建可视化的知识图谱。平台集成了数据处理、聚类分析、可视化展示、人工标注与知识图谱生成等功能，助力于深入理解和分析大模型的内部结构与功能分布。

---

# 🛠 Installation

The model partitioning and annotation platform is based on PyTorch and the OpenMMLab ecosystem. Some pre-trained weights are from timm.

## 1. Create Python Environment

```bash
conda create -n open-mmlab python=3.8 pytorch=2.1.0 cudatoolkit=11.8 torchvision -c pytorch -y
conda activate open-mmlab
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or, install key dependencies manually:

```bash
pip install torch==2.1.0
pip install mmcv==2.1.0
pip install mmdet==3.3.0
pip install mmengine-lite==0.10.7
pip install py2neo==2021.2.3
pip install pandas==1.3.5
pip install numpy==1.21.6
pip install timm
```

> **Note:** The code requires `torch.fx` for computational graph extraction. Please ensure your PyTorch version is >=1.10.

## 3. Configure Neo4j

Set up a Neo4j database (version 5.25.1 recommended) and note the connection details for later use.

---

# 🚀 Getting Started

To use the platform, follow these main steps:

## 1. Model Feature Extraction & Similarity Computation

- Specify your dataset and model configuration.
- Extract model feature embeddings and compute representation similarity.

Example:

```bash
PYTHONPATH="$PWD" python similarity/get_rep.py \
  $Config_file \
  --out $Feature_path \
  [--checkpoint $Checkpoint]
```
- All feature embeddings should be saved as `.pth` files in the same directory.
- Compute feature similarity (results saved as `net1.net2.pkl`):

```bash
PYTHONPATH="$PWD" python similarity/compute_sim.py \
  --feat_path $Feat_directory \
  --sim_func $Similarity_function   # [cka, rbf_cka, lr]
```
- Compute feature size (input-output dimensions):

```bash
PYTHONPATH="$PWD" python similarity/count_inout_size.py \
  --root $Feat_directory
```
- The result is a JSON file (e.g., `MODEL_INOUT_SHAPE.json`).

## 2. Network Partitioning

- Perform clustering-based partitioning with parameters:
  - Number of partitions `K` (default: 3)
  - Number of initializations `n_init` (default: 10)
  - Maximum iterations `max_iter` (default: 300)

Example:

```bash
PYTHONPATH="$PWD" python similarity/partition.py \
  --sim_path $Feat_similarity_path \
  --K $Num_partition \
  --trial $Num_repeat_runs \
  --eps $Size_ratio_each_block \
  --num_iter $Maximum_num_iter_eachrun
```
- The output is an assignment file in `.pkl` format.

## 3. Heatmap Generation & Annotation

- Select a `.pkl` file from the `features` directory to generate a heatmap.
- Upload the generated heatmap to the platform for functional annotation.

## 4. Knowledge Graph Construction

- Export the annotation results as a CSV file.
- Import the CSV into Neo4j to generate a knowledge graph.

---

For more details, please refer to the code and scripts in the `similarity/` and `features/` directories.

## Platform Demo

![Platform Demo](frontend/platform.png)

