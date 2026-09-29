#!/usr/bin/env python3
"""
在虚拟机上使用 ONNX 构建 text2vec FAISS 索引
"""

import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def build_text2vec_index(data_dir, batch_size=64, n_workers=4):
    """使用 ONNX + 多线程构建 FAISS 索引"""
    
    print("=" * 60)
    print("Text2Vec FAISS 索引构建工具 (虚拟机版)")
    print("=" * 60)
    
    import faiss
    import onnxruntime as ort
    
    onnx_path = f"{data_dir}/text2vec-small-chinese_optimized.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = f"{data_dir}/text2vec-small-chinese.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = os.path.expanduser("~/桌面/ai/models_onnx/text2vec-small-chinese.onnx")
    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 模型不存在")
        return False
    
    print(f"加载 ONNX 模型: {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    encoder = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    print("ONNX 编码器加载成功")
    
    try:
        from tokenizers import Tokenizer
        tokenizer_path = f"{data_dir}/tokenizer.json"
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.expanduser("~/桌面/ai/models_onnx/tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            use_native_tokenizer = True
            print("原生 Tokenizer 加载成功")
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("shibing624/text2vec-base-chinese")
            use_native_tokenizer = False
            print("Transformers Tokenizer 加载成功")
    except Exception as e:
        print(f"Tokenizer 加载失败: {e}")
        return False
    
    json_path = f"{data_dir}/sbert_768_full.json"
    if not os.path.exists(json_path):
        json_path = os.path.expanduser("~/桌面/ai/voice_assistant/rk3588_deploy/full_data/sbert_768_full.json")
    if not os.path.exists(json_path):
        print(f"错误: 数据文件不存在")
        return False
    
    print(f"\n加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogues = data['dialogues']
    questions = [d['question'] for d in dialogues]
    n_samples = len(questions)
    print(f"共 {n_samples} 条数据")
    
    def encode_text(text):
        """编码单个文本"""
        if use_native_tokenizer:
            encoding = tokenizer.encode(text)
            ids = encoding.ids[:128]
            mask = encoding.attention_mask[:128]
            if len(ids) < 128:
                ids = ids + [0] * (128 - len(ids))
                mask = mask + [0] * (128 - len(mask))
            input_ids = np.array([ids], dtype=np.int64)
            attention_mask = np.array([mask], dtype=np.int64)
        else:
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')
            input_ids = encoded['input_ids'].astype(np.int64)
            attention_mask = encoded['attention_mask'].astype(np.int64)
        
        outputs = encoder.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
        last_hidden = outputs[0]
        
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_emb = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(attention_mask.astype(np.float32), axis=1, keepdims=True), a_min=1e-9, a_max=None)
        embedding = sum_emb / sum_mask
        
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embedding = embedding / norms
        
        return embedding[0]
    
    print(f"\n开始编码 (batch_size={batch_size}, n_workers={n_workers})...")
    start_time = time.time()
    all_embeddings = []
    
    for i in range(0, n_samples, batch_size):
        batch_texts = questions[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            batch_embeddings = list(executor.map(encode_text, batch_texts))
        
        all_embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % 1000 == 0 or i + batch_size >= n_samples:
            elapsed = time.time() - start_time
            progress = min(i + batch_size, n_samples)
            speed = progress / elapsed
            eta = (n_samples - progress) / speed if speed > 0 else 0
            print(f"  进度: {progress}/{n_samples} ({progress/n_samples*100:.1f}%) | 速度: {speed:.1f}/s | ETA: {eta:.0f}s")
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"\n编码完成, 形状: {embeddings.shape}, 耗时: {time.time()-start_time:.1f}s")
    
    print("\n构建 FAISS 索引...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    output_path = f"{data_dir}/text2vec_768_full_vector.faiss"
    faiss.write_index(index, output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"索引保存: {output_path}")
    print(f"索引大小: {size_mb:.1f} MB")
    
    print("\n完成!")
    return True


if __name__ == "__main__":
    data_dir = "/home/ubuntu/桌面/ai/voice_assistant/rk3588_deploy/full_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    build_text2vec_index(data_dir)
