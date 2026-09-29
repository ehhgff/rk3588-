#!/usr/bin/env python3
"""
构建科室分类索引
将向量索引按科室分片，减少搜索范围
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict

def build_dept_index(data_dir="/userdata/medical_rag_full"):
    """构建科室分类索引"""
    print("=" * 60)
    print("构建科室分类索引")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/3] 加载医疗数据...")
    with open(f"{data_dir}/sbert_768_full.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogues = data['dialogues']
    print(f"  共 {len(dialogues)} 条对话")
    
    # 按科室分组
    print("\n[2/3] 按科室分组...")
    dept_docs = defaultdict(list)  # 科室 -> [doc_id列表]
    dept_count = defaultdict(int)
    
    for doc_id, d in enumerate(dialogues):
        dept = d.get('department', '未知')
        dept_docs[dept].append(doc_id)
        dept_count[dept] += 1
    
    print(f"  共 {len(dept_docs)} 个科室")
    for dept, count in sorted(dept_count.items(), key=lambda x: -x[1]):
        print(f"    {dept}: {count}条")
    
    # 加载向量索引
    print("\n[3/3] 构建科室向量索引...")
    import faiss
    
    # 加载完整索引
    index_path = f"{data_dir}/text2vec_768_full_vector.faiss"
    if not os.path.exists(index_path):
        index_path = f"{data_dir}/sbert_768_full_vector.faiss"
    
    full_index = faiss.read_index(index_path)
    vector_dim = full_index.d
    print(f"  加载完整索引: {full_index.ntotal} 条, {vector_dim} 维")
    
    # 为每个科室创建子索引
    dept_indices = {}
    dept_doc_maps = {}  # 子索引位置 -> 原始doc_id
    
    for dept, doc_ids in dept_docs.items():
        # 提取该科室的所有向量
        vectors = []
        for doc_id in doc_ids:
            vec = full_index.reconstruct(int(doc_id))
            vectors.append(vec)
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # 创建FAISS索引
        dept_index = faiss.IndexFlatIP(vector_dim)
        dept_index.add(vectors)
        
        dept_indices[dept] = dept_index
        dept_doc_maps[dept] = doc_ids
        
        print(f"  {dept}: {len(doc_ids)} 条 -> 子索引 {dept_index.ntotal}")
    
    # 保存科室索引
    print("\n[4/4] 保存索引文件...")
    index_data = {
        'dept_indices': dept_indices,
        'dept_doc_maps': dept_doc_maps,
        'dept_count': dict(dept_count),
        'vector_dim': vector_dim
    }
    
    output_path = f"{data_dir}/dept_vector_index.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  保存到: {output_path}")
    print(f"  文件大小: {file_size:.1f} MB")
    
    print("\n✓ 科室分类索引构建完成!")
    return index_data

if __name__ == "__main__":
    build_dept_index()
