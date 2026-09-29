#!/usr/bin/env python3
"""
RK3588 交互式医疗RAG系统
支持用户输入查询并获取结果
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/userdata/medical_rag')

class InteractiveRAG:
    """交互式RAG系统"""
    
    def __init__(self):
        self.dialogues = []
        self.titles = []
        self.questions = []
        self.answers = []
        self.departments = []
        self.vector_index = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.loaded = False
        
    def load(self):
        """加载所有资源"""
        if self.loaded:
            return True
            
        print("=" * 70)
        print("加载医疗RAG系统...")
        print("=" * 70)
        
        # 加载数据
        print("\n[1/3] 加载医疗数据...")
        data_path = "/userdata/medical_rag/sbert_768_final.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.dialogues = data['dialogues']
        for d in self.dialogues:
            self.titles.append(d.get('title', ''))
            self.questions.append(d.get('question', ''))
            self.answers.append(d.get('answer', ''))
            self.departments.append(d.get('department', '未知'))
        
        print(f"  ✓ 加载完成: {len(self.dialogues)} 条对话")
        
        # 加载向量索引
        print("\n[2/3] 加载FAISS向量索引...")
        import faiss
        index_path = "/userdata/medical_rag/sbert_768_final_vector.faiss"
        self.vector_index = faiss.read_index(index_path)
        print(f"  ✓ 索引加载完成: {self.vector_index.ntotal} 向量")
        
        # 加载BM25索引
        print("\n[3/3] 加载BM25索引...")
        import pickle
        bm25_path = "/userdata/medical_rag/sbert_768_final_bm25.pkl"
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
        self.bm25 = data['bm25']
        self.tokenized_corpus = data['tokenized_corpus']
        print(f"  ✓ BM25加载完成")
        
        self.loaded = True
        print("\n" + "=" * 70)
        print("系统加载完成! 可以开始查询")
        print("=" * 70)
        return True
    
    def encode_query(self, query_text):
        """编码查询 (使用随机向量模拟SBERT)"""
        # 实际应用中应该使用SBERT模型编码
        # 这里使用随机向量作为演示
        np.random.seed(hash(query_text) % 2**32)
        return np.random.random((1, 768)).astype('float32')
    
    def search(self, query_text, k=3):
        """搜索"""
        if not self.loaded:
            print("系统未加载，请先调用load()")
            return []
        
        # 编码查询
        query_vector = self.encode_query(query_text)
        
        # 向量搜索
        start = time.time()
        vec_scores, vec_indices = self.vector_index.search(query_vector, k*2)
        vec_time = time.time() - start
        
        # BM25搜索
        import jieba
        tokens = list(jieba.cut(query_text.lower()))
        tokens = [t for t in tokens if t.strip()]
        
        start = time.time()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_time = time.time() - start
        
        # 获取top-k BM25结果
        bm25_top_k = np.argsort(bm25_scores)[-k*2:][::-1]
        
        # RRF融合
        combined = {}
        for rank, idx in enumerate(vec_indices[0]):
            if idx >= 0:
                combined[idx] = combined.get(idx, 0) + 0.6 / (rank + 1)
        
        for rank, idx in enumerate(bm25_top_k):
            combined[idx] = combined.get(idx, 0) + 0.4 / (rank + 1)
        
        # 排序
        sorted_results = sorted(combined.items(), key=lambda x: -x[1])
        final_indices = [idx for idx, _ in sorted_results[:k]]
        
        total_time = vec_time + bm25_time
        
        # 构建结果
        results = []
        for idx in final_indices:
            results.append({
                'index': idx,
                'department': self.departments[idx],
                'title': self.titles[idx],
                'question': self.questions[idx],
                'answer': self.answers[idx][:200] + '...' if len(self.answers[idx]) > 200 else self.answers[idx]
            })
        
        return results, total_time
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 70)
        print("交互式医疗咨询系统")
        print("=" * 70)
        print("\n提示:")
        print("  - 输入您的问题")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'stats' 查看系统统计")
        print("=" * 70)
        
        while True:
            print()
            query = input("请输入问题: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n再见!")
                break
            
            if query.lower() == 'stats':
                self.show_stats()
                continue
            
            # 执行搜索
            print(f"\n搜索: {query}")
            print("-" * 70)
            
            results, elapsed = self.search(query, k=3)
            
            print(f"检索时间: {elapsed*1000:.2f}ms")
            print(f"找到结果: {len(results)} 条\n")
            
            for i, r in enumerate(results, 1):
                print(f"[{i}] [{r['department']}] {r['title']}")
                print(f"    问题: {r['question'][:60]}...")
                print(f"    回答: {r['answer'][:80]}...")
                print()
    
    def show_stats(self):
        """显示统计信息"""
        print("\n" + "=" * 70)
        print("系统统计")
        print("=" * 70)
        print(f"数据规模: {len(self.dialogues)} 条对话")
        print(f"向量维度: 768")
        print(f"向量索引: {self.vector_index.ntotal} 向量")
        print(f"索引类型: IndexFlatIP")
        
        # 科室分布
        dept_counts = {}
        for dept in self.departments:
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        print(f"\n科室分布 (前10):")
        for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {dept}: {count}条")

def main():
    """主函数"""
    rag = InteractiveRAG()
    
    # 加载资源
    if not rag.load():
        print("✗ 系统加载失败")
        return 1
    
    # 进入交互模式
    rag.interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
