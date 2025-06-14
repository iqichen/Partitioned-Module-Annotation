from py2neo import Graph, Node, Relationship
import pandas as pd

def rebuild_knowledge_graph():
    # 1. 连接到Neo4j
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "1a2b3c4d5e"))
    
    # 清空现有数据
    print("正在清空数据库...")
    graph.run("MATCH (n) DETACH DELETE n")

    # 2. 读取CSV数据
    csv_path = r"E:\Code\Partitioned Module Annotation\neo4j\annotation.csv"
    df = pd.read_csv(csv_path)

    # 3. 节点缓存和辅助函数
    node_cache = {}


    def get_or_create_node(label, name):
        key = f"{label}_{name}"
        if key not in node_cache:
            node = Node(label, name=name)
            graph.create(node)
            node_cache[key] = node
        return node_cache[key]


    def get_or_create_module(model_name, module_name):
        key = f"Module_{model_name}_{module_name}"
        if key not in node_cache:
            module = Node("Module", name=module_name, belongs_to_model=model_name)
            graph.create(module)
            node_cache[key] = module
        return node_cache[key]


    # 4. 创建基础节点和关系
    print("正在创建节点和关系...")
    for _, row in df.iterrows():
        try:
            model = get_or_create_node("Model", row.iloc[0])
            module = get_or_create_module(row.iloc[0], row.iloc[1])
            category = get_or_create_node("Category", row.iloc[2])
            function = get_or_create_node("Function", row.iloc[4])

            # 模型-模块关系
            if not graph.exists(Relationship(model, "CONTAINS", module)):
                rel_mm = Relationship(model, "CONTAINS", module)
                rel_mm["width"] = "1px"
                rel_mm["color"] = "gray"
                graph.create(rel_mm)

            # 模块-类别关系（临时存储响应值）
            rel_mc = Relationship(module, "响应值", category)
            rel_mc["response_level"] = float(row.iloc[3])
            rel_mc["width"] = "1px"
            rel_mc["color"] = "gray"
            rel_mc["caption"] = f"{row.iloc[3]}"
            graph.create(rel_mc)

            # 类别-功能关系（临时存储置信度）
            rel_cf = Relationship(category, "置信度", function)
            rel_cf["confidence"] = float(row.iloc[5])
            rel_cf["width"] = "1px"
            rel_cf["color"] = "gray"
            rel_cf["caption"] = f"{row.iloc[5]}"
            graph.create(rel_cf)

        except Exception as e:
            print(f"Error in row {_}: {e}")

    # 5. 标记高亮关系（关键步骤）
    highlight_queries = [
        # 标记每个category下响应值最高的module关系
        """
        MATCH (m:Module)-[r:响应值]->(c:Category)
        WITH c, MAX(r.response_level) AS max_response
        MATCH (m:Module)-[r:响应值]->(c:Category)
        WHERE r.response_level = max_response
        SET r.color = "#FF0000", r.width = "3px", r.caption = "最高响应值: " + toString(r.response_level)
        RETURN count(r) AS highlighted_responses
        """,

        # 标记每个module下置信度最高的function关系
        """
        MATCH (c:Category)-[r:置信度]->(f:Function)
        WITH c, MAX(r.confidence) AS max_confidence
        MATCH (c:Category)-[r:置信度]->(f:Function)
        WHERE r.confidence = max_confidence
        SET r.color = "#00FF00", r.width = "3px", r.caption = "最高置信度: " + toString(r.confidence)
        RETURN count(r) AS highlighted_confidences
        """
    ]

    results = []
    for query in highlight_queries:
        results.append(graph.run(query).data())
    
    print("已标记关系：")
    print(f"最大响应值：{results[0][0]['highlighted_responses']}")
    print(f"最高置信度：{results[1][0]['highlighted_confidences']}")

    print("知识图谱构建完成！")

# 执行知识图谱构建
if __name__ == "__main__":
    rebuild_knowledge_graph()