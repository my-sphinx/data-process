"""
数据采样Pipeline - 从400万log中筛选10000条高质量训练数据

架构设计:
1. 数据清洗 - 去除脏数据、无效数据
2. 向量化嵌入 - 使用sentence-transformers
3. 向量数据库 - ChromaDB存储和检索
4. 聚类采样 - K-means保证多样性
5. 质量评分 - 多维度打分筛选
6. 人工标注准备 - 输出待标注数据

硬件要求:
- 内存: 16GB+（处理400万数据）
- GPU: 可选，用于加速embedding（无GPU也可运行）
- 磁盘: 10GB+（存储向量数据库）
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ===============================================
# 第1步: 数据清洗模块
# ===============================================
class DataCleaner:
    """
    数据清洗器 - 去除脏数据、无效数据
    
    清洗规则:
    1. 长度过滤: 3-500字符
    2. 编码清洗: 移除乱码
    3. 重复过滤: 精确去重
    4. 格式验证: 基本文本格式检查
    5. 敏感词过滤: 广告、垃圾信息
    """
    
    def __init__(self):
        # 定义无效模式
        self.invalid_patterns = [
            r'^[\s\W]*$',  # 纯空白或特殊字符
            r'.*[\x00-\x1F].*',  # 控制字符
            r'.*[\uFFFD].*',  # Unicode替换字符(乱码标志)
        ]
        
        # 垃圾信息关键词（示例，根据实际情况调整）
        self.spam_keywords = [
            '广告', '推广', '加微信', 'VX', '扫码', '点击链接',
            'http://', 'https://', 'www.', '.com', '.cn',
            '￥', '$$$', '免费领取', '限时优惠'
        ]
    
    def is_valid_length(self, text: str, min_len: int = 3, max_len: int = 500) -> bool:
        """检查文本长度是否合理"""
        return min_len <= len(text.strip()) <= max_len
    
    def is_valid_format(self, text: str) -> bool:
        """检查文本格式是否有效"""
        for pattern in self.invalid_patterns:
            if re.match(pattern, text):
                return False
        return True
    
    def contains_spam(self, text: str) -> bool:
        """检查是否包含垃圾信息"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.spam_keywords)
    
    def clean_text(self, text: str) -> str:
        """清洗文本（标准化）"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空白
        text = text.strip()
        return text
    
    def filter_batch(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        批量过滤文本
        
        返回:
        - cleaned_texts: 清洗后的有效文本列表
        - valid_indices: 有效文本的原始索引
        """
        cleaned_texts = []
        valid_indices = []
        
        for idx, text in enumerate(tqdm(texts, desc="数据清洗")):
            # 跳过空值
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            # 清洗文本
            text = self.clean_text(text)
            
            # 应用过滤规则
            if not self.is_valid_length(text):
                continue
            if not self.is_valid_format(text):
                continue
            if self.contains_spam(text):
                continue
            
            cleaned_texts.append(text)
            valid_indices.append(idx)
        
        return cleaned_texts, valid_indices


# ===============================================
# 第2步: 向量嵌入模块
# ===============================================
class TextEmbedder:
    """
    文本向量化器 - 使用sentence-transformers
    
    模型选择:
    - 中文: 'moka-ai/m3e-base' (更好的中文支持)
    - 备选: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    """
    
    def __init__(self, model_name: str = "moka-ai/m3e-base", device: str = "cuda"):
        """
        初始化嵌入模型
        
        参数:
        - model_name: 模型名称或本地路径
        - device: 'cuda' 或 'cpu'
        """
        print(f"加载嵌入模型: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            
            # 自动检测设备
            if device == "cuda" and not torch.cuda.is_available():
                print("警告: CUDA不可用，切换到CPU模式")
                device = "cpu"
            
            # 处理本地路径（兼容 Path 对象和字符串）
            # 如果是Path对象，转换为字符串
            if hasattr(model_name, '__fspath__'):  # Path对象
                model_name = str(model_name)
                print(f"检测到Path对象，已转换为字符串")
            
            if os.path.exists(model_name):
                # 转换为绝对路径并标准化
                model_name = os.path.abspath(model_name)
                print(f"检测到本地模型路径: {model_name}")
                
                # 验证必要文件
                required_files = ['config.json', 'pytorch_model.bin']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_name, f))]
                
                if missing_files:
                    print(f"⚠️ 警告: 缺少文件 {missing_files}")
                    print(f"请确保模型目录包含所有必要文件")
            
            # 确保是字符串类型（critical for SentenceTransformer）
            model_name = str(model_name)
            
            # 加载模型
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
            print(f"✓ 模型加载成功，使用设备: {device}")
            
        except Exception as e:
            print(f"错误: 无法加载模型 {model_name}")
            print(f"详细错误: {str(e)}")
            print(f"\n可能的原因:")
            print(f"1. 模型文件不完整，请运行: python download_models.py")
            print(f"2. 路径错误，请检查路径是否正确")
            print(f"3. 缺少依赖，请运行: pip install sentence-transformers torch")
            raise e
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量生成文本嵌入
        
        参数:
        - texts: 文本列表
        - batch_size: 批次大小（根据显存调整）
        
        返回:
        - embeddings: (N, D) 嵌入矩阵
        """
        print(f"生成嵌入向量 (batch_size={batch_size})...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化，便于计算相似度
        )
        
        return embeddings


# ===============================================
# 第3步: 向量数据库模块
# ===============================================
class VectorDatabase:
    """
    向量数据库 - 使用ChromaDB
    
    功能:
    1. 存储文本嵌入
    2. 相似度检索
    3. 去重（移除高度相似的样本）
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化ChromaDB
        
        参数:
        - persist_directory: 数据库持久化目录
        """
        print(f"初始化向量数据库: {persist_directory}")
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            # 创建或获取集合
            self.collection = self.client.get_or_create_collection(
                name="log_data",
                metadata={"description": "Log数据向量存储"}
            )
            
            print(f"✓ 数据库初始化成功，当前文档数: {self.collection.count()}")
            
        except Exception as e:
            print(f"错误: 无法初始化ChromaDB")
            print(f"请安装: pip install chromadb")
            raise e
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict] = None
    ):
        """
        批量添加文档到向量数据库
        
        参数:
        - texts: 文本列表
        - embeddings: 嵌入矩阵
        - metadata: 元数据列表（可选）
        """
        print(f"添加 {len(texts)} 个文档到向量数据库...")
        
        # 生成ID
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        # 准备元数据
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        
        # 批量添加（分批避免内存问题）
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc="写入数据库"):
            batch_end = min(i + batch_size, len(texts))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadata[i:batch_end]
            )
        
        print(f"✓ 数据库写入完成，总文档数: {self.collection.count()}")
    
    def deduplicate(self, similarity_threshold: float = 0.95) -> List[str]:
        """
        去重 - 移除高度相似的文档
        
        参数:
        - similarity_threshold: 相似度阈值（0-1）
        
        返回:
        - unique_ids: 去重后的文档ID列表
        """
        print(f"执行去重，相似度阈值: {similarity_threshold}")
        
        # 简单策略：遍历所有文档，查找每个文档的最近邻
        # 如果相似度超过阈值，则标记为重复
        
        all_docs = self.collection.get(include=["embeddings", "documents"])
        n_docs = len(all_docs["ids"])
        
        # 标记重复文档
        duplicates = set()
        
        for i in tqdm(range(n_docs), desc="查找重复"):
            if all_docs["ids"][i] in duplicates:
                continue
            
            # 查询最近邻
            results = self.collection.query(
                query_embeddings=[all_docs["embeddings"][i]],
                n_results=10  # 查找前10个最相似的
            )
            
            # 检查相似度（ChromaDB返回距离，需转换为相似度）
            for j, distance in enumerate(results["distances"][0]):
                # 余弦相似度 = 1 - 余弦距离
                similarity = 1 - distance
                
                if similarity > similarity_threshold:
                    duplicate_id = results["ids"][0][j]
                    if duplicate_id != all_docs["ids"][i]:
                        duplicates.add(duplicate_id)
        
        print(f"✓ 发现 {len(duplicates)} 个重复文档")
        
        # 返回非重复文档ID
        unique_ids = [doc_id for doc_id in all_docs["ids"] if doc_id not in duplicates]
        return unique_ids


# ===============================================
# 第4步: 聚类采样模块
# ===============================================
class DiversitySampler:
    """
    多样性采样器 - 基于K-means聚类
    
    策略:
    1. K-means聚类将数据分为K类
    2. 从每类中按比例采样
    3. 保证覆盖所有主题/模式
    """
    
    def __init__(self, n_clusters: int = 100, random_state: int = 42):
        """
        初始化聚类采样器
        
        参数:
        - n_clusters: 聚类数量（建议100-500）
        - random_state: 随机种子
        """
        from sklearn.cluster import MiniBatchKMeans
        
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=1000,  # Mini-batch加速
            verbose=1
        )
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        聚类并预测每个样本的类别
        
        参数:
        - embeddings: (N, D) 嵌入矩阵
        
        返回:
        - labels: (N,) 类别标签
        """
        print(f"执行K-means聚类 (k={self.n_clusters})...")
        labels = self.kmeans.fit_predict(embeddings)
        print(f"✓ 聚类完成")
        
        # 统计每类样本数
        unique, counts = np.unique(labels, return_counts=True)
        print(f"聚类分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
        
        return labels
    
    def stratified_sample(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 10000,
        strategy: str = "balanced",
        min_per_cluster: int = 5,
        max_per_cluster: int = None,
        frequencies: List[int] = None  # 新增：频率信息用于加权采样
    ) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
        """
        改进的分层采样 - 确保每个聚类都有代表，保证多样性和完整性
        
        参数:
        - texts: 文本列表
        - embeddings: 嵌入矩阵
        - labels: 聚类标签
        - n_samples: 目标采样数量
        - strategy: 采样策略
            - 'balanced': 每类采样相同数量
            - 'proportional': 按类别比例采样
            - 'hybrid': 混合策略（保证最小值+按比例）
        - min_per_cluster: 每个簇最少采样数量（保证完整性）
        - max_per_cluster: 每个簇最多采样数量（防止某些簇过度代表）
        - frequencies: 文本频率列表（可选）。如果提供，将用于加权采样，高频样本更可能被选中
        
        返回:
        - sampled_texts: 采样后的文本
        - sampled_embeddings: 采样后的嵌入
        - sampled_indices: 采样的原始索引
        - sampling_stats: 采样统计信息
        """
        print(f"分层采样 (目标: {n_samples}条, 策略: {strategy}, 每簇最少: {min_per_cluster})...")
        
        # 统计每个簇的样本数
        unique_labels, cluster_counts = np.unique(labels, return_counts=True)
        n_active_clusters = len(unique_labels)
        
        print(f"活跃簇数: {n_active_clusters}/{self.n_clusters}")
        print(f"簇大小分布: min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
        
        # 频率感知采样
        use_frequency_weighting = frequencies is not None
        if use_frequency_weighting:
            frequencies_array = np.array(frequencies, dtype=float)
            # 使用sqrt平滑，避免超高频样本占据过多权重
            freq_weights = np.sqrt(frequencies_array)
            print(f"✓ 启用频率感知采样 (高频样本更可能被选中)")
        
        # ===== 第一阶段：确保每个簇至少采样 min_per_cluster 条 =====
        sampled_indices = []
        cluster_sample_counts = {}  # 记录每个簇采样了多少条
        
        print("阶段1: 确保每个簇的最小采样量...")
        for cluster_id in unique_labels:
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            # 确定该簇的最小采样数
            n_min = min(min_per_cluster, len(cluster_indices))
            
            if n_min > 0:
                # 如果有频率信息，使用加权采样
                if use_frequency_weighting:
                    cluster_weights = freq_weights[cluster_indices]
                    cluster_weights = cluster_weights / cluster_weights.sum()  # 归一化
                    selected = np.random.choice(
                        cluster_indices,
                        size=n_min,
                        replace=False,
                        p=cluster_weights
                    )
                else:
                    selected = np.random.choice(
                        cluster_indices,
                        size=n_min,
                        replace=False
                    )
                sampled_indices.extend(selected)
                cluster_sample_counts[cluster_id] = n_min
        
        print(f"  已采样: {len(sampled_indices)} 条 (最小保证)")
        
        # ===== 第二阶段：根据策略分配剩余配额 =====
        remaining_quota = n_samples - len(sampled_indices)
        
        if remaining_quota > 0:
            print(f"阶段2: 分配剩余 {remaining_quota} 条配额...")
            
            # 计算每个簇还能采样多少
            cluster_allocation = {}
            
            for cluster_id in unique_labels:
                cluster_mask = (labels == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]
                cluster_size = len(cluster_indices)
                already_sampled = cluster_sample_counts.get(cluster_id, 0)
                available = cluster_size - already_sampled
                
                if available <= 0:
                    cluster_allocation[cluster_id] = 0
                    continue
                
                # 根据策略计算分配量
                if strategy == "balanced":
                    # 平均分配剩余配额
                    target = remaining_quota // n_active_clusters
                elif strategy == "proportional":
                    # 按簇大小比例分配
                    target = int(remaining_quota * cluster_size / len(texts))
                else:  # hybrid
                    # 混合：保证平均+按比例调整
                    base_allocation = remaining_quota // n_active_clusters
                    proportion_bonus = int((cluster_size / len(texts)) * remaining_quota * 0.3)
                    target = base_allocation + proportion_bonus
                
                # 限制在可用范围内
                target = min(target, available)
                
                # 应用最大值限制
                if max_per_cluster is not None:
                    target = min(target, max_per_cluster - already_sampled)
                
                cluster_allocation[cluster_id] = max(0, target)
            
            # 标准化分配（确保总和不超过剩余配额）
            total_allocated = sum(cluster_allocation.values())
            if total_allocated > remaining_quota:
                scale_factor = remaining_quota / total_allocated
                cluster_allocation = {
                    cid: int(count * scale_factor) 
                    for cid, count in cluster_allocation.items()
                }
            
            # 执行第二阶段采样
            for cluster_id, n_additional in cluster_allocation.items():
                if n_additional <= 0:
                    continue
                
                cluster_mask = (labels == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]
                
                # 排除已采样的
                available_indices = list(set(cluster_indices) - set(sampled_indices))
                
                if len(available_indices) > 0:
                    n_sample = min(n_additional, len(available_indices))
                    
                    # 如果有频率信息，使用加权采样
                    if use_frequency_weighting:
                        available_weights = freq_weights[available_indices]
                        available_weights = available_weights / available_weights.sum()
                        selected = np.random.choice(
                            available_indices,
                            size=n_sample,
                            replace=False,
                            p=available_weights
                        )
                    else:
                        selected = np.random.choice(
                            available_indices,
                            size=n_sample,
                            replace=False
                        )
                    sampled_indices.extend(selected)
                    cluster_sample_counts[cluster_id] = cluster_sample_counts.get(cluster_id, 0) + n_sample
            
            print(f"  已采样: {len(sampled_indices)} 条 (含第二阶段)")
        
        # ===== 第三阶段：如果仍不足目标，从大簇补充 =====
        if len(sampled_indices) < n_samples:
            remaining = n_samples - len(sampled_indices)
            print(f"阶段3: 补充剩余 {remaining} 条...")
            
            # 按簇大小排序，从大到小补充
            sorted_clusters = sorted(
                unique_labels,
                key=lambda cid: np.sum(labels == cid),
                reverse=True
            )
            
            for cluster_id in sorted_clusters:
                if remaining <= 0:
                    break
                
                cluster_mask = (labels == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]
                available_indices = list(set(cluster_indices) - set(sampled_indices))
                
                if len(available_indices) > 0:
                    n_sample = min(remaining, len(available_indices))
                    
                    # 如果有频率信息，使用加权采样
                    if use_frequency_weighting:
                        available_weights = freq_weights[available_indices]
                        available_weights = available_weights / available_weights.sum()
                        selected = np.random.choice(
                            available_indices,
                            size=n_sample,
                            replace=False,
                            p=available_weights
                        )
                    else:
                        selected = np.random.choice(
                            available_indices,
                            size=n_sample,
                            replace=False
                        )
                    sampled_indices.extend(selected)
                    cluster_sample_counts[cluster_id] = cluster_sample_counts.get(cluster_id, 0) + n_sample
                    remaining -= n_sample
        
        # 转换为numpy数组
        sampled_indices = np.array(sampled_indices[:n_samples])
        
        # ===== 生成采样统计 =====
        sampling_stats = {
            'total_samples': len(sampled_indices),
            'n_active_clusters': n_active_clusters,
            'n_covered_clusters': len(cluster_sample_counts),
            'coverage_rate': len(cluster_sample_counts) / n_active_clusters,
            'samples_per_cluster': cluster_sample_counts,
            'min_samples_per_cluster': min(cluster_sample_counts.values()) if cluster_sample_counts else 0,
            'max_samples_per_cluster': max(cluster_sample_counts.values()) if cluster_sample_counts else 0,
            'mean_samples_per_cluster': np.mean(list(cluster_sample_counts.values())) if cluster_sample_counts else 0,
        }
        
        print(f"\n✓ 采样完成: {len(sampled_indices)} 条")
        print(f"  簇覆盖率: {sampling_stats['coverage_rate']:.1%} ({sampling_stats['n_covered_clusters']}/{n_active_clusters})")
        print(f"  每簇采样: min={sampling_stats['min_samples_per_cluster']}, "
              f"max={sampling_stats['max_samples_per_cluster']}, "
              f"mean={sampling_stats['mean_samples_per_cluster']:.1f}")
        
        # 提取采样结果
        sampled_texts = [texts[i] for i in sampled_indices]
        sampled_embeddings = embeddings[sampled_indices]
        
        return sampled_texts, sampled_embeddings, sampled_indices, sampling_stats


# ===============================================
# 第5步: 质量评分模块
# ===============================================
class QualityScorer:
    """
    质量评分器 - 多维度评估文本质量
    
    评分维度:
    1. 信息密度 - 是否包含实质内容
    2. 语法完整性 - 是否为完整句子
    3. 领域相关性 - 是否与目标领域相关
    4. 可标注性 - 是否适合人工标注
    """
    
    def __init__(self, domain_keywords: List[str] = None):
        """
        初始化质量评分器
        
        参数:
        - domain_keywords: 领域关键词列表（寿险相关）
        """
        if domain_keywords is None:
            # 寿险领域关键词
            domain_keywords = [
                '寿险', '终身寿险', '定期寿险', '保险', '保障',
                '理赔', '受益人', '保额', '保费', '投保',
                '保单', '身故', '全残', '责任', '现金价值'
            ]
        
        self.domain_keywords = domain_keywords
    
    def score_information_density(self, text: str) -> float:
        """
        信息密度得分 (0-1)
        
        规则:
        - 字数适中: 10-200字 (1.0分)
        - 包含实词: 名词、动词比例
        """
        word_count = len(text)
        
        # 字数得分
        if 10 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 10:
            length_score = word_count / 10
        else:
            length_score = max(0.5, 1 - (word_count - 200) / 300)
        
        # 实词比例（简单启发式：中文字符比例）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        char_ratio = chinese_chars / max(len(text), 1)
        
        return (length_score + char_ratio) / 2
    
    def score_grammar_completeness(self, text: str) -> float:
        """
        语法完整性得分 (0-1)
        
        规则:
        - 有标点符号: +0.3
        - 以问号结尾: +0.3 (问句)
        - 没有连续特殊字符: +0.4
        """
        score = 0.0
        
        # 包含标点
        if re.search(r'[，。！？、；：]', text):
            score += 0.3
        
        # 以问号结尾（问句更适合意图识别）
        if text.endswith('？') or text.endswith('?'):
            score += 0.3
        
        # 没有连续特殊字符
        if not re.search(r'[^\u4e00-\u9fffa-zA-Z0-9]{3,}', text):
            score += 0.4
        
        return min(score, 1.0)
    
    def score_domain_relevance(self, text: str) -> float:
        """
        领域相关性得分 (0-1)
        
        规则:
        - 包含领域关键词数量
        """
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in text)
        
        # 包含1个关键词: 0.5, 包含2个: 0.75, 包含3+: 1.0
        if keyword_count == 0:
            return 0.0
        elif keyword_count == 1:
            return 0.5
        elif keyword_count == 2:
            return 0.75
        else:
            return 1.0
    
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量评分
        
        返回:
        - scores: (N,) 综合得分数组
        """
        print("计算质量得分...")
        
        scores = []
        for text in tqdm(texts, desc="质量评分"):
            density = self.score_information_density(text)
            grammar = self.score_grammar_completeness(text)
            relevance = self.score_domain_relevance(text)
            
            # 综合得分（加权平均）
            # 领域相关性最重要，其次是信息密度和语法
            final_score = (
                0.4 * relevance +
                0.3 * density +
                0.3 * grammar
            )
            
            scores.append(final_score)
        
        scores = np.array(scores)
        print(f"✓ 得分统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
        
        return scores


# ===============================================
# 主Pipeline
# ===============================================
class SamplingPipeline:
    """
    完整的数据采样Pipeline
    
    流程:
    1. 加载原始数据
    2. 数据清洗
    3. 向量嵌入
    4. 向量数据库存储
    5. 去重
    6. 聚类采样
    7. 质量评分与筛选
    8. 输出待标注数据
    """
    
    def __init__(
        self,
        input_file: str,
        output_file: str = "data/sampled_for_annotation.csv",
        n_target_samples: int = 10000,
        n_clusters: int = 200,
        embedding_model: str = "moka-ai/m3e-base"
    ):
        """
        初始化Pipeline
        
        参数:
        - input_file: 输入文件路径（400万条log）
        - output_file: 输出文件路径（10000条待标注数据）
        - n_target_samples: 目标采样数量
        - n_clusters: 聚类数量
        - embedding_model: 嵌入模型名称
        """
        self.input_file = input_file
        self.output_file = output_file
        self.n_target_samples = n_target_samples
        self.n_clusters = n_clusters
        self.embedding_model = embedding_model
        
        # 初始化各模块
        self.cleaner = DataCleaner()
        self.embedder = None  # 延迟加载
        self.vector_db = None  # 延迟加载
        self.sampler = None  # 延迟加载
        self.scorer = QualityScorer()
    
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        假设数据格式:
        - CSV文件，包含'text'列（或其他列名）
        """
        print(f"加载数据: {self.input_file}")
        
        # 根据文件类型加载
        if self.input_file.endswith('.csv'):
            # 尝试多种编码和方式加载CSV
            encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1', 'iso-8859-1']
            df = None
            
            # 先尝试不同编码
            for encoding in encodings_to_try:
                try:
                    print(f"尝试编码: {encoding}")
                    df = pd.read_csv(
                        self.input_file,
                        on_bad_lines='skip',  # 跳过错误行
                        encoding=encoding,
                        engine='python'  # 使用更宽容的Python引擎
                    )
                    print(f"✓ 数据加载完成（编码: {encoding}）: {len(df)} 条")
                    break
                except UnicodeDecodeError as e:
                    print(f"  编码 {encoding} 失败: {str(e)[:80]}")
                    continue
                except Exception as e:
                    print(f"  编码 {encoding} 出错: {str(e)[:80]}")
                    continue
            
            # 如果所有编码都失败，尝试忽略错误
            if df is None:
                print(f"⚠️ 所有标准编码失败，尝试忽略编码错误...")
                try:
                    df = pd.read_csv(
                        self.input_file,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        encoding_errors='ignore',  # 忽略编码错误
                        engine='python'
                    )
                    print(f"✓ 数据加载完成（忽略编码错误）: {len(df)} 条")
                except Exception as e:
                    print(f"⚠️ 忽略编码错误也失败: {str(e)[:100]}")
                    
                    # 最后尝试兼容旧版本
                    try:
                        df = pd.read_csv(
                            self.input_file,
                            error_bad_lines=False,
                            warn_bad_lines=False,
                            encoding='gbk',
                            engine='python'
                        )
                        print(f"✓ 数据加载完成（兼容模式+GBK）: {len(df)} 条")
                    except Exception as e2:
                        print(f"⚠️ 兼容模式失败: {str(e2)[:100]}")
                        print(f"尝试最宽容模式...")
                        
                        # 最后手段: 逐行读取
                        df = self._load_csv_tolerant(self.input_file)
                        print(f"✓ 数据加载完成（宽容模式）: {len(df)} 条")
                    
        elif self.input_file.endswith('.json'):
            df = pd.read_json(self.input_file, lines=True)
        elif self.input_file.endswith('.parquet'):
            df = pd.read_parquet(self.input_file)
        else:
            raise ValueError(f"不支持的文件格式: {self.input_file}")
        
        print(f"列名: {df.columns.tolist()}")
        
        return df
    
    def _load_csv_tolerant(self, filepath: str) -> pd.DataFrame:
        """
        最宽容的CSV加载方式（逐行处理）
        
        当标准方法失败时使用
        """
        import csv
        
        print(f"使用逐行读取模式（较慢但最稳定）...")
        
        # 尝试多种编码
        encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            rows = []
            error_count = 0
            
            try:
                print(f"  尝试编码: {encoding}")
                with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                    # 尝试检测CSV格式
                    sample = f.read(1024)
                    f.seek(0)
                    
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                    except:
                        dialect = csv.excel  # 使用默认
                    
                    reader = csv.reader(f, dialect)
                    
                    # 读取表头
                    try:
                        headers = next(reader)
                    except StopIteration:
                        raise ValueError("文件为空")
                    
                    # 逐行读取
                    for line_num, row in enumerate(reader, start=2):
                        try:
                            if len(row) == len(headers):
                                rows.append(row)
                            else:
                                # 列数不匹配，尝试修正
                                if len(row) < len(headers):
                                    row.extend([''] * (len(headers) - len(row)))
                                else:
                                    row = row[:len(headers)]
                                rows.append(row)
                        except Exception as e:
                            error_count += 1
                            if error_count <= 10:  # 只打印前10个错误
                                print(f"    跳过行 {line_num}: {str(e)[:50]}")
                
                # 如果成功读取了数据，跳出循环
                if len(rows) > 0:
                    print(f"  ✓ 使用编码 {encoding} 成功读取")
                    if error_count > 0:
                        print(f"  ⚠️ 跳过了 {error_count} 个错误行")
                    
                    # 构建DataFrame
                    df = pd.DataFrame(rows, columns=headers)
                    return df
                    
            except Exception as e:
                print(f"  编码 {encoding} 失败: {str(e)[:80]}")
                continue
        
        # 如果所有编码都失败
        raise ValueError(f"无法使用任何编码读取文件: {filepath}")
    
    def run(self):
        """运行完整Pipeline"""
        
        print("=" * 70)
        print("数据采样Pipeline启动")
        print(f"输入: {self.input_file}")
        print(f"目标: 从400万条中筛选 {self.n_target_samples} 条")
        print("=" * 70)
        
        # ========== 步骤1: 加载数据 ==========
        df = self.load_data()
        
        # 假设文本列名为'text'或'query'或'question'
        text_column = None
        for col in ['text', 'query', 'question', 'content', 'message']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print(f"错误: 未找到文本列，请指定列名")
            print(f"可用列: {df.columns.tolist()}")
            return
        
        print(f"使用文本列: '{text_column}'")
        texts = df[text_column].tolist()
        
        # ========== 步骤2: 数据清洗 ==========
        print("\n" + "=" * 70)
        print("步骤1: 数据清洗")
        print("=" * 70)
        
        cleaned_texts, valid_indices = self.cleaner.filter_batch(texts)
        
        print(f"✓ 清洗完成: {len(cleaned_texts)} / {len(texts)} ({len(cleaned_texts)/len(texts)*100:.1f}%)")
        
        # 如果清洗后数据量仍然很大，可以先随机采样到100万
        if len(cleaned_texts) > 1_000_000:
            print(f"数据量较大({len(cleaned_texts)}条)，先随机采样到100万条以加速处理...")
            sample_indices = np.random.choice(
                len(cleaned_texts),
                size=1_000_000,
                replace=False
            )
            cleaned_texts = [cleaned_texts[i] for i in sample_indices]
            valid_indices = [valid_indices[i] for i in sample_indices]
        
        # ========== 步骤3: 文本去重（保留频率信息）==========
        print("\n" + "=" * 70)
        print("步骤2: 去重并统计频率")
        print("=" * 70)
        
        # 统计频率
        from collections import Counter
        text_freq = Counter(cleaned_texts)
        
        # 创建DataFrame
        df_with_freq = pd.DataFrame({'text': cleaned_texts, 'original_index': valid_indices})
        
        # 去重但保留频率信息
        unique_df = df_with_freq.drop_duplicates(subset=['text'], keep='first').copy()
        unique_df['frequency'] = unique_df['text'].map(text_freq)
        
        # 频率统计
        freq_values = unique_df['frequency'].values
        print(f"✓ 去重完成: {len(unique_df)} 条唯一文本")
        print(f"  频率分布: min={freq_values.min()}, max={freq_values.max()}, mean={freq_values.mean():.1f}, median={np.median(freq_values):.1f}")
        print(f"  高频样本(>=10次): {np.sum(freq_values >= 10)} 条 ({np.sum(freq_values >= 10)/len(freq_values):.1%})")
        print(f"  中频样本(2-9次): {np.sum((freq_values >= 2) & (freq_values < 10))} 条")
        print(f"  低频样本(1次): {np.sum(freq_values == 1)} 条 ({np.sum(freq_values == 1)/len(freq_values):.1%})")
        print(f"  总频率覆盖: {freq_values.sum():,} 次原始查询")
        
        # 去重率分析
        dedup_rate = len(unique_df) / len(df_with_freq)
        print(f"  去重率: {dedup_rate:.1%}")
        
        if dedup_rate < 0.7:
            print(f"  💡 去重率较低，说明有大量重复查询，建议使用频率感知采样")
        
        cleaned_texts = unique_df['text'].tolist()
        valid_indices = unique_df['original_index'].tolist()
        text_frequencies = unique_df['frequency'].tolist()
        
        # ========== 步骤4: 向量嵌入 ==========
        print("\n" + "=" * 70)
        print("步骤3: 生成文本嵌入")
        print("=" * 70)
        
        self.embedder = TextEmbedder(model_name=self.embedding_model)
        embeddings = self.embedder.embed_batch(cleaned_texts, batch_size=64)
        
        print(f"✓ 嵌入生成完成: shape={embeddings.shape}")
        
        # ========== 步骤5: 向量数据库存储（可选，用于语义去重）==========
        # 注意：对于100万+数据，向量数据库去重会很慢
        # 这里跳过，或者只对最终采样后的数据做语义去重
        
        # ========== 步骤6: K-means聚类采样 ==========
        print("\n" + "=" * 70)
        print("步骤4: K-means聚类 + 多样性采样")
        print("=" * 70)
        
        # 先采样到3倍目标数量，留给质量评分筛选
        # 对于98% F1目标，建议最终数据量15000-20000条
        n_intermediate_samples = self.n_target_samples * 3
        
        self.sampler = DiversitySampler(n_clusters=self.n_clusters)
        labels = self.sampler.fit_predict(embeddings)
        
        # 使用改进的分层采样（确保每个簇都有代表，考虑频率权重）
        sampled_texts, sampled_embeddings, sampled_indices, sampling_stats = self.sampler.stratified_sample(
            texts=cleaned_texts,
            embeddings=embeddings,
            labels=labels,
            n_samples=n_intermediate_samples,
            strategy="hybrid",  # 混合策略：保证多样性+按比例
            min_per_cluster=5,  # 每簇至少5条
            max_per_cluster=n_intermediate_samples // self.n_clusters * 3,  # 防止单簇过多
            frequencies=text_frequencies  # 新增：传入频率信息用于加权采样
        )
        
        # 提取采样数据的频率
        sampled_frequencies = [text_frequencies[i] for i in sampled_indices]
        
        # 检查簇覆盖率
        if sampling_stats['coverage_rate'] < 0.95:
            print(f"\n⚠️ 警告: 簇覆盖率仅 {sampling_stats['coverage_rate']:.1%}，可能影响多样性")
            print(f"建议: 增加目标采样数量或减少聚类数")
        else:
            print(f"✓ 簇覆盖率优秀: {sampling_stats['coverage_rate']:.1%}")
        
        # ========== 步骤7: 质量评分与筛选（保持簇平衡）==========
        print("\n" + "=" * 70)
        print("步骤5: 质量评分与簇平衡筛选")
        print("=" * 70)
        
        scores = self.scorer.score_batch(sampled_texts)
        
        # 获取采样数据的簇标签
        sampled_labels = labels[sampled_indices]
        
        # 簇平衡筛选：确保最终数据仍保持簇多样性
        print("执行簇平衡筛选（保证质量的同时维持多样性）...")
        
        final_indices = []
        cluster_final_counts = {}
        
        # 计算每个簇的目标数量（按采样比例）
        unique_sampled_labels = np.unique(sampled_labels)
        samples_per_cluster_target = {}
        
        for cluster_id in unique_sampled_labels:
            cluster_count_in_sample = np.sum(sampled_labels == cluster_id)
            # 按比例分配最终数量，但至少保留3条
            target = max(3, int(self.n_target_samples * cluster_count_in_sample / len(sampled_labels)))
            samples_per_cluster_target[cluster_id] = target
        
        # 标准化（确保总和不超过目标）
        total_target = sum(samples_per_cluster_target.values())
        if total_target > self.n_target_samples:
            scale = self.n_target_samples / total_target
            samples_per_cluster_target = {
                cid: max(3, int(count * scale))
                for cid, count in samples_per_cluster_target.items()
            }
        
        # 从每个簇中选择最高质量的样本
        for cluster_id in unique_sampled_labels:
            cluster_mask = (sampled_labels == cluster_id)
            cluster_sample_indices = np.where(cluster_mask)[0]
            cluster_scores = scores[cluster_sample_indices]
            
            # 按质量排序
            sorted_cluster_indices = cluster_sample_indices[np.argsort(cluster_scores)[::-1]]
            
            # 取该簇的目标数量
            n_take = min(samples_per_cluster_target[cluster_id], len(sorted_cluster_indices))
            selected = sorted_cluster_indices[:n_take]
            
            final_indices.extend(selected)
            cluster_final_counts[cluster_id] = n_take
        
        # 如果数量不足，从高质量样本中补充
        if len(final_indices) < self.n_target_samples:
            remaining = self.n_target_samples - len(final_indices)
            
            # 找到未被选中的样本
            all_indices = set(range(len(sampled_texts)))
            selected_set = set(final_indices)
            remaining_indices = list(all_indices - selected_set)
            
            # 按质量排序补充
            remaining_scores = scores[remaining_indices]
            top_remaining = [remaining_indices[i] for i in np.argsort(remaining_scores)[::-1][:remaining]]
            final_indices.extend(top_remaining)
        
        # 截断到目标数量
        final_indices = final_indices[:self.n_target_samples]
        
        final_texts = [sampled_texts[i] for i in final_indices]
        final_scores = scores[final_indices]
        final_labels = sampled_labels[final_indices]
        final_original_indices = [valid_indices[sampled_indices[i]] for i in final_indices]
        final_frequencies = [sampled_frequencies[i] for i in final_indices]
        
        # 验证最终簇覆盖
        final_unique_clusters = len(np.unique(final_labels))
        final_coverage_rate = final_unique_clusters / len(unique_sampled_labels)
        
        print(f"✓ 最终筛选: {len(final_texts)} 条")
        print(f"  质量得分: {final_scores.min():.2f} - {final_scores.max():.2f} (mean: {final_scores.mean():.2f})")
        print(f"  最终簇覆盖: {final_unique_clusters}/{len(unique_sampled_labels)} = {final_coverage_rate:.1%}")
        
        if final_coverage_rate < 0.90:
            print(f"⚠️ 警告: 质量筛选后簇覆盖率下降到 {final_coverage_rate:.1%}")
            print(f"建议: 降低质量阈值或增加中间采样数量")
        
        # ========== 步骤8: 输出待标注数据 ==========
        print("\n" + "=" * 70)
        print("步骤6: 保存待标注数据")
        print("=" * 70)
        
        output_df = pd.DataFrame({
            'text': final_texts,
            'frequency': final_frequencies,  # 新增：保留频率信息
            'quality_score': final_scores,
            'cluster_id': final_labels,
            'original_index': final_original_indices,
            'importance': ['高频' if f >= 10 else '中频' if f >= 2 else '低频' for f in final_frequencies],  # 新增：重要性标签
            'label': '',  # 空白列，等待人工标注
        })
        
        # 创建输出目录
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存
        output_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存到: {self.output_file}")
        
        # ========== 保存详细统计信息 ==========
        # 计算多样性指标
        cluster_distribution = {}
        for cluster_id in np.unique(final_labels):
            count = np.sum(final_labels == cluster_id)
            cluster_distribution[int(cluster_id)] = int(count)
        
        # Shannon熵（衡量簇分布的均匀性）
        cluster_counts = np.array(list(cluster_distribution.values()))
        cluster_probs = cluster_counts / cluster_counts.sum()
        shannon_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
        max_entropy = np.log(len(cluster_counts))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        stats = {
            # 数据流统计
            'pipeline_stats': {
                'total_input': len(texts),
                'after_cleaning': len(cleaned_texts),
                'after_dedup': len(cleaned_texts),
                'after_clustering_sample': len(sampled_texts),
                'final_output': len(final_texts),
                'retention_rate': len(final_texts) / len(texts),
            },
            
            # 聚类统计
            'clustering_stats': {
                'n_clusters': self.n_clusters,
                'n_active_clusters': int(sampling_stats['n_active_clusters']),
                'n_covered_clusters_initial': int(sampling_stats['n_covered_clusters']),
                'n_covered_clusters_final': int(final_unique_clusters),
                'coverage_rate_initial': float(sampling_stats['coverage_rate']),
                'coverage_rate_final': float(final_coverage_rate),
            },
            
            # 质量统计
            'quality_stats': {
                'score_min': float(final_scores.min()),
                'score_max': float(final_scores.max()),
                'score_mean': float(final_scores.mean()),
                'score_std': float(final_scores.std()),
                'score_median': float(np.median(final_scores)),
                'score_q25': float(np.percentile(final_scores, 25)),
                'score_q75': float(np.percentile(final_scores, 75)),
            },
            
            # 多样性统计
            'diversity_stats': {
                'cluster_distribution': cluster_distribution,
                'samples_per_cluster_min': int(cluster_counts.min()),
                'samples_per_cluster_max': int(cluster_counts.max()),
                'samples_per_cluster_mean': float(cluster_counts.mean()),
                'samples_per_cluster_std': float(cluster_counts.std()),
                'shannon_entropy': float(shannon_entropy),
                'normalized_entropy': float(normalized_entropy),
                'entropy_interpretation': '接近1表示分布均匀，接近0表示分布集中'
            },
            
            # 建议
            'recommendations': self._generate_recommendations(
                final_coverage_rate=final_coverage_rate,
                normalized_entropy=normalized_entropy,
                mean_quality=final_scores.mean(),
                n_samples=len(final_texts)
            )
        }
        
        stats_file = self.output_file.replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, indent=2, fp=f, ensure_ascii=False)
        
        print(f"✓ 统计信息已保存到: {stats_file}")
        
        # ========== 打印总结报告 ==========
        print("\n" + "=" * 70)
        print("数据质量与多样性报告")
        print("=" * 70)
        print(f"\n📊 数据规模:")
        print(f"  原始数据: {len(texts):,} 条")
        print(f"  清洗后: {len(cleaned_texts):,} 条")
        print(f"  最终输出: {len(final_texts):,} 条")
        
        print(f"\n🎯 簇覆盖率:")
        print(f"  初始采样: {sampling_stats['coverage_rate']:.1%} ({sampling_stats['n_covered_clusters']}/{sampling_stats['n_active_clusters']})")
        print(f"  最终数据: {final_coverage_rate:.1%} ({final_unique_clusters}/{len(unique_sampled_labels)})")
        
        print(f"\n⭐ 质量评分:")
        print(f"  平均分: {final_scores.mean():.3f}")
        print(f"  范围: {final_scores.min():.3f} - {final_scores.max():.3f}")
        
        print(f"\n🔥 频率分布 (原始查询重要性):")
        final_freq_array = np.array(final_frequencies)
        total_freq = final_freq_array.sum()
        print(f"  采样覆盖总频率: {total_freq:,} 次原始查询")
        print(f"  平均频率: {final_freq_array.mean():.1f}")
        print(f"  高频样本(>=10次): {np.sum(final_freq_array >= 10)} 条 ({np.sum(final_freq_array >= 10)/len(final_freq_array):.1%})")
        print(f"  中频样本(2-9次): {np.sum((final_freq_array >= 2) & (final_freq_array < 10))} 条")
        print(f"  低频样本(1次): {np.sum(final_freq_array == 1)} 条 ({np.sum(final_freq_array == 1)/len(final_freq_array):.1%})")
        
        print(f"\n🌈 多样性指标:")
        print(f"  标准化熵: {normalized_entropy:.3f} (1.0为最佳)")
        print(f"  每簇样本: {cluster_counts.min()}-{cluster_counts.max()} (平均: {cluster_counts.mean():.1f})")
        
        print(f"\n💡 建议:")
        for rec in stats['recommendations']:
            print(f"  • {rec}")
        
        # ========== 完成 ==========
        print("\n" + "=" * 70)
        print("✓ Pipeline完成！")
        print("=" * 70)
        print(f"\n📝 下一步:")
        print(f"1. 打开文件: {self.output_file}")
        print(f"2. 对 'label' 列进行人工标注（寿险相关 / 拒识）")
        print(f"3. 标注完成后，使用标注数据训练模型")
        print(f"4. 查看详细统计: {stats_file}")
        print("=" * 70)
    
    def _generate_recommendations(
        self,
        final_coverage_rate: float,
        normalized_entropy: float,
        mean_quality: float,
        n_samples: int
    ) -> List[str]:
        """
        基于统计指标生成建议
        
        目标: 98% F1 for 1.7B-8B LLM微调
        """
        recommendations = []
        
        # 簇覆盖率建议
        if final_coverage_rate < 0.90:
            recommendations.append(
                f"⚠️ 簇覆盖率较低({final_coverage_rate:.1%})，可能影响模型泛化能力。"
                f"建议增加采样数量或降低质量阈值。"
            )
        elif final_coverage_rate >= 0.95:
            recommendations.append(
                f"✓ 簇覆盖率优秀({final_coverage_rate:.1%})，数据多样性良好。"
            )
        
        # 多样性建议
        if normalized_entropy < 0.7:
            recommendations.append(
                f"⚠️ 数据分布集中(熵={normalized_entropy:.2f})，某些簇可能过度代表。"
                f"建议使用'balanced'策略或增加min_per_cluster。"
            )
        elif normalized_entropy >= 0.85:
            recommendations.append(
                f"✓ 数据分布均匀(熵={normalized_entropy:.2f})，多样性优秀。"
            )
        
        # 质量建议
        if mean_quality < 0.4:
            recommendations.append(
                f"⚠️ 平均质量较低({mean_quality:.2f})，可能影响训练效果。"
                f"建议加强数据清洗或调整质量评分权重。"
            )
        elif mean_quality >= 0.6:
            recommendations.append(
                f"✓ 平均质量良好({mean_quality:.2f})。"
            )
        
        # 数据量建议（针对98% F1目标）
        if n_samples < 10000:
            recommendations.append(
                f"⚠️ 数据量较少({n_samples}条)，对于1.7B-8B模型可能不足。"
                f"建议增加到15000-20000条以达到98% F1目标。"
            )
        elif n_samples >= 15000:
            recommendations.append(
                f"✓ 数据量充足({n_samples}条)，适合1.7B-8B模型微调。"
            )
        else:
            recommendations.append(
                f"数据量({n_samples}条)适中，可考虑增加到15000-20000条以优化性能。"
            )
        
        # 总体建议
        if final_coverage_rate >= 0.95 and normalized_entropy >= 0.8 and mean_quality >= 0.5:
            recommendations.append(
                "🎉 数据质量与多样性均达标，可以开始标注和训练！"
            )
        
        # 针对98% F1的具体建议
        recommendations.append(
            "💡 达到98% F1的关键因素："
        )
        recommendations.append(
            "  1. 确保每个意图类别至少有50-100条标注样本"
        )
        recommendations.append(
            "  2. 对边界case进行重点标注（模糊、歧义样本）"
        )
        recommendations.append(
            "  3. 使用主动学习：先训练，找出低置信度样本，补充标注"
        )
        recommendations.append(
            "  4. 考虑数据增强：同义替换、回译等"
        )
        
        return recommendations


# ===============================================
# 命令行入口
# ===============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="数据采样Pipeline - 确保多样性和完整性，达到98% F1微调目标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础用法（10000条样本）
  python data_sampling_pipeline.py --input data/logs.csv
  
  # 针对98% F1目标（推荐15000-20000条）
  python data_sampling_pipeline.py --input data/logs.csv --n_samples 15000 --n_clusters 300
  
  # 使用本地嵌入模型
  python data_sampling_pipeline.py --input data/logs.csv --embedding_model ./models/m3e-base
  
  # 确保每个簇至少10条样本
  python data_sampling_pipeline.py --input data/logs.csv --n_samples 20000 --n_clusters 200 --min_per_cluster 10
        """
    )
    
    # 基础参数
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径（400万条log数据，支持CSV/JSON/Parquet）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sampled_for_annotation.csv",
        help="输出文件路径 (默认: data/sampled_for_annotation.csv)"
    )
    
    # 采样参数
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="目标采样数量 (建议15000-20000以达到98%% F1) (默认: 10000)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=200,
        help="K-means聚类数量 (影响多样性，建议200-500) (默认: 200)"
    )
    
    # 多样性参数
    parser.add_argument(
        "--min_per_cluster",
        type=int,
        default=5,
        help="每个簇最少采样数量，确保完整性 (默认: 5)"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="hybrid",
        choices=["balanced", "proportional", "hybrid"],
        help="采样策略: balanced(均衡), proportional(按比例), hybrid(混合) (默认: hybrid)"
    )
    
    # 模型参数
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="moka-ai/m3e-base",
        help="嵌入模型名称或本地路径 (默认: moka-ai/m3e-base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="运行设备 (默认: cuda，自动降级到cpu)"
    )
    
    args = parser.parse_args()
    
    print(f"""
{'='*70}
数据采样Pipeline配置
{'='*70}
输入文件: {args.input}
输出文件: {args.output}
目标样本数: {args.n_samples}
聚类数量: {args.n_clusters}
每簇最少: {args.min_per_cluster}
采样策略: {args.sampling_strategy}
嵌入模型: {args.embedding_model}
{'='*70}
""")
    
    # 运行Pipeline
    pipeline = SamplingPipeline(
        input_file=args.input,
        output_file=args.output,
        n_target_samples=args.n_samples,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model
    )
    
    pipeline.run()

