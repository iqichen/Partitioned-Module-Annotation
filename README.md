# 大模型模块化语义标注平台

## 项目简介

本项目旨在对大型语言模型进行模块拆分，并对各个模块进行语义标注，最终通过知识图谱进行存储和可视化。整个流程通过一个集成的前端平台进行操作，实现了大模型能力的细粒度分析与理解。

## Quick Start

### Environment Requirements

- Python 3.8.12
- CUDA 11.8
- PyTorch 2.1.0
- MMCV 2.1.0
- MMDetection 3.3.0
- MMEngine-lite 0.10.7
- Neo4j 5.25.1
- JDK 17.0.8

### Installation

1. Clone the repository
```bash
git clone git@github.com:iqichen/Partitioned-Module-Annotation.git
cd Partitioned-Module-Annotation
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure Neo4j database connection

4. Run the application
```bash
# To be added
```

## 平台展示

![平台展示图](placeholder_for_platform_image.png)

## 主要功能

- **模型模块拆分**：将大型语言模型按功能和结构拆分为多个子模块
- **语义标注**：对拆分后的模块进行语义级别的标注和分类
- **知识图谱构建**：基于Neo4j构建知识图谱，存储模块间的关系和语义信息
- **前端交互平台**：提供友好的用户界面，支持模型分析、标注和可视化全流程操作

## 技术栈

- **后端**：Python, Neo4j图数据库
- **前端**：Web界面（待补充具体技术）
- **数据处理**：Pandas等数据分析库

## 项目结构

```
Partitioned-Module-Annotation/
├── neo4j/                # Neo4j相关脚本和数据
│   ├── import.py         # 知识图谱构建脚本
│   └── annotation.csv    # 标注数据
├── frontend/             # 前端代码
└── ...
```

## 贡献指南

欢迎提交问题和功能需求！如果您想贡献代码，请先fork本仓库，然后提交pull request。

## 许可证

[待补充]

## 更新日志

最后更新：2025年6月14日 