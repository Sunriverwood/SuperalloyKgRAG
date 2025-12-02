Plan: 无监督图推理 + 推荐系统实现（修订版）
基于已有的 final_graph.json 知识图谱和 embedding.db 向量数据库，实现一个query-aware的图推理系统，支持多跳路径推理、节点推荐和可解释的推理链条生成。系统采用自监督训练方式，不依赖人工标注，通过图结构学习和伪查询任务提升推理能力。利用图谱关系中已有的 weight 和 composite_importance 综合评分参数优化推理质量。
Steps
创建数据加载模块 [core/reasoning/data_loader.py]：加载 final_graph.json 构建 NetworkX 图对象，从 embedding.db 的 entities 和 relationships 表提取节点/边向量，提取边的 weight 和 composite_importance 作为初始权重，构建邻接表/稀疏矩阵和 adjacency mask，统一 embedding 维度
实现 Query-Aware RGAT 图编码器 [core/reasoning/models/rgat.py]：设计关系图注意力网络类 QueryAwareRGAT，实现 forward(x_v, edge_index, edge_type, edge_weights, query_emb) 方法，将 composite_importance 融入 attention 计算作为先验权重，使用 attention 机制融合邻居信息并应用图约束 mask，支持多层消息传递
构建自监督训练模块 [core/reasoning/training/trainer.py]：实现三类自监督任务 - (a) 边重构 Link Prediction 使用 weight 作为正样本权重，(b) 图对比学习 InfoNCE，(c) 伪查询-尾实体匹配任务，设计 Trainer 类管理训练循环、损失组合、checkpoint 保存
开发推理引擎 [core/reasoning/inference/reasoner.py]：实现 GraphReasoner 类，提供 (1) 节点打分 score_nodes(query) 使用训练好的匹配函数，(2) Personalized PageRank 多跳传播 propagate_ppr(query, alpha=0.15) 转移矩阵使用 composite_importance 归一化，(3) 路径抽取 extract_paths(start_nodes, end_nodes, max_depth=3) 基于图约束的 BFS 搜索并使用 weight 和注意力得分计算路径概率
集成到查询系统 [core/query_qwen/reasoning_query_qwen.py]：创建 ReasoningQueryHandler 类继承现有查询接口，封装完整推理流程：查询编码 → 节点检索 → 图推理传播 → 路径生成 → LLM 答案合成，支持可解释输出，路径展示包含 composite_importance 评分
配置与工具函数 [utils/graph_reasoning_utils.py, utils/reasoning_config.py]：实现图约束工具 apply_adjacency_mask()，路径评分 score_path_by_importance(path, G) 结合 weight 和 attention，边权重预处理 normalize_edge_weights(G, use_composite=True)，伪查询生成器 PseudoQueryGenerator，扩展 settings.yaml 添加 reasoning 参数节（模型路径、PPR alpha、路径深度等）
Further Considerations
综合评分的使用策略：在 PPR 转移矩阵构建时，使用 composite_importance 归一化得到边转移概率 P_uv = composite_importance_uv / Σ_{v'} composite_importance_uv'；在路径评分时，路径概率计算为 P(π|q) = I(π) × ∏ α_uv^(q) × weight_uv，同时考虑 attention 和原始权重
依赖库选择：推荐使用 PyTorch + PyTorch Geometric (PyG) 实现 RGAT，或 DGL 作为替代；使用现有 LanceDB 功能做 ANN 检索；路径搜索使用 NetworkX 现有图结构；权重处理复用 utils/community_importance.py 的设计模式
分阶段实现顺序：建议先实现数据加载和边权重提取 → RGAT 基础结构集成 composite_importance → 单任务自监督训练(边重构使用加权损失) → 验证训练收敛 → 添加伪查询任务 → 实现 PPR 推理使用综合评分 → 路径抽取并展示权重 → 最终集成到查询系统
权重归一化与平滑：由于 composite_importance 可能分布不均，在 utils/graph_reasoning_utils.py 中实现归一化函数支持多种策略（min-max、softmax、log-scale）；在 PPR 中添加平滑项避免零权重边被完全忽略
可解释性增强：路径输出时展示每条边的三元组信息、composite_importance 评分、query-aware attention 权重和最终路径概率，便于用户理解推理过程；在日志中记录权重分布统计辅助调试