#!/usr/bin/env python3
"""
FAISS索引压缩工具
使用IVF1024,PQ32配置压缩234MB索引到~25MB
"""

import faiss
import numpy as np
import json
import time
import os
import sys

def load_original_data(data_dir="/userdata/medical_rag_full"):
    """加载原始数据"""
    print("[1/4] 加载原始数据...")
    
    # 加载JSON数据
    with open(f"{data_dir}/sbert_768_full.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogues = data['dialogues']
    print(f"  加载 {len(dialogues)} 条医疗对话")
    
    # 提取向量
    vectors = []
    for d in dialogues:
        emb = d.get('embedding')
        if emb:
            vectors.append(emb)
    
    vectors = np.array(vectors).astype('float32')
    print(f"  向量维度: {vectors.shape}")
    
    return vectors, dialogues

def create_compressed_index(vectors, nlist=1024, m=32):
    """创建IVF-PQ压缩索引"""
    print("\n[2/4] 创建IVF1024,PQ32压缩索引...")
    
    dim = vectors.shape[1]  # 768
    
    # 创建量化器
    quantizer = faiss.IndexFlatIP(dim)
    
    # 创建IVF-PQ索引
    # IVF1024: 1024个倒排列表
    # PQ32: 将768维分成32个子向量，每子向量8bit
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    
    print(f"  索引类型: IVF{nlist},PQ{m}")
    print(f"  向量维度: {dim}")
    print(f"  倒排列表数: {nlist}")
    print(f"  PQ子向量数: {m}")
    
    # 训练索引
    print("\n  训练索引...")
    start = time.time()
    
    # 使用10%数据训练
    n_train = min(int(len(vectors) * 0.1), 50000)
    train_vectors = vectors[:n_train]
    
    index.train(train_vectors)
    print(f"  训练完成，耗时: {time.time()-start:.1f}s")
    
    # 添加向量
    print("\n  添加向量到索引...")
    start = time.time()
    index.add(vectors)
    print(f"  添加完成，耗时: {time.time()-start:.1f}s")
    
    return index

def test_accuracy(original_index_path, compressed_index, test_queries, dialogues, k=5):
    """测试压缩后的精度"""
    print("\n[3/4] 测试压缩索引精度...")
    
    # 加载原始索引
    original_index = faiss.read_index(original_index_path)
    
    # 测试样本
    test_indices = np.random.choice(len(dialogues), min(100, len(dialogues)), replace=False)
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for idx in test_indices:
        query_vector = np.array([dialogues[idx].get('embedding')]).astype('float32')
        
        # 原始索引搜索
        D_orig, I_orig = original_index.search(query_vector, k)
        
        # 压缩索引搜索
        D_comp, I_comp = compressed_index.search(query_vector, k)
        
        # 检查Top-1
        if I_orig[0][0] == I_comp[0][0]:
            correct_top1 += 1
        
        # 检查Top-5
        if len(set(I_orig[0]) & set(I_comp[0])) > 0:
            correct_top5 += 1
        
        total += 1
    
    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100
    
    print(f"  Top-1 准确率: {top1_acc:.1f}%")
    print(f"  Top-5 准确率: {top5_acc:.1f}%")
    print(f"  测试样本数: {total}")
    
    return top1_acc, top5_acc

def save_compressed_index(index, output_path):
    """保存压缩索引"""
    print("\n[4/4] 保存压缩索引...")
    
    faiss.write_index(index, output_path)
    
    # 获取文件大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  保存路径: {output_path}")
    print(f"  文件大小: {size_mb:.1f}MB")
    
    return size_mb

def main():
    print("=" * 60)
    print("FAISS索引压缩工具")
    print("配置: IVF1024,PQ32")
    print("=" * 60)
    
    data_dir = "/userdata/medical_rag_full"
    output_path = f"{data_dir}/sbert_768_full_vector_compressed.faiss"
    original_path = f"{data_dir}/sbert_768_full_vector.faiss"
    
    # 检查原始索引
    if not os.path.exists(original_path):
        print(f"错误: 原始索引不存在: {original_path}")
        sys.exit(1)
    
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    print(f"\n原始索引大小: {original_size:.1f}MB")
    
    # 加载数据
    vectors, dialogues = load_original_data(data_dir)
    
    # 创建压缩索引
    index = create_compressed_index(vectors, nlist=1024, m=32)
    
    # 测试精度
    top1_acc, top5_acc = test_accuracy(original_path, index, None, dialogues)
    
    # 保存索引
    compressed_size = save_compressed_index(index, output_path)
    
    # 总结
    print("\n" + "=" * 60)
    print("压缩结果汇总")
    print("=" * 60)
    print(f"原始大小:     {original_size:.1f}MB")
    print(f"压缩后大小:   {compressed_size:.1f}MB")
    print(f"压缩率:       {original_size/compressed_size:.1f}x")
    print(f"Top-1准确率:  {top1_acc:.1f}%")
    print(f"Top-5准确率:  {top5_acc:.1f}%")
    print("=" * 60)
    
    if top1_acc >= 90:
        print("\n✓ 压缩成功，精度满足要求！")
    else:
        print("\n⚠ 精度较低，建议调整参数或保留原始索引")
    
    # 建议
    print("\n使用建议:")
    print(f"  1. 测试压缩索引: python3 -c \"import faiss; idx=faiss.read_index('{output_path}'); print('加载成功')\"")
    print(f"  2. 备份原始索引: cp {original_path} {original_path}.backup")
    print(f"  3. 替换原始索引: cp {output_path} {original_path}")

if __name__ == '__main__':
    main()
