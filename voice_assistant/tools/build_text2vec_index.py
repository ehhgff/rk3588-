#!/usr/bin/env python3
"""
使用 text2vec 模型构建 FAISS 索引
用于 RKNN 加速的 RAG 服务
"""

import os
import sys
import json
import time
import numpy as np

def build_text2vec_index(data_dir, batch_size=32):
    """使用 text2vec 构建 FAISS 索引"""
    
    print("=" * 60)
    print("Text2Vec FAISS 索引构建工具")
    print("=" * 60)
    
    import faiss
    
    try:
        from rknnlite.api import RKNNLite
        use_rknn = True
        print("使用 RKNN NPU 加速")
    except ImportError:
        use_rknn = False
        print("RKNN 不可用，使用 ONNX Runtime")
    
    if use_rknn:
        # 优先使用 FP16 高精度模型
        rknn_path = f"{data_dir}/text2vec-small-chinese_fp16.rknn"
        if not os.path.exists(rknn_path):
            rknn_path = f"{data_dir}/text2vec-small-chinese_v2.rknn"
        if not os.path.exists(rknn_path):
            print(f"错误: RKNN 模型不存在")
            return False
        print(f"使用 RKNN 模型: {rknn_path}")
        
        encoder = RKNNLite(verbose=False)
        ret = encoder.load_rknn(rknn_path)
        if ret != 0:
            print("RKNN 模型加载失败")
            return False
        ret = encoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print("RKNN 运行时初始化失败")
            return False
        print("RKNN 编码器加载成功")
    else:
        import onnxruntime as ort
        onnx_path = f"{data_dir}/text2vec-small-chinese.onnx"
        if not os.path.exists(onnx_path):
            print(f"错误: ONNX 模型不存在: {onnx_path}")
            return False
        encoder = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("ONNX 编码器加载成功")
    
    try:
        from tokenizers import Tokenizer
        tokenizer_path = f"{data_dir}/tokenizer.json"
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
        print(f"错误: 数据文件不存在: {json_path}")
        return False
    
    print(f"\n加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dialogues = data['dialogues']
    questions = [d['question'] for d in dialogues]
    n_samples = len(questions)
    print(f"共 {n_samples} 条数据")
    
    print(f"\n开始编码 (batch_size={batch_size})...")
    all_embeddings = []
    
    for i in range(0, n_samples, batch_size):
        batch_texts = questions[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch_texts:
            if use_native_tokenizer:
                encoding = tokenizer.encode(text)
                input_ids = np.array([encoding.ids[:128]], dtype=np.int64)
                if input_ids.shape[1] < 128:
                    pad = np.zeros((1, 128 - input_ids.shape[1]), dtype=np.int64)
                    input_ids = np.concatenate([input_ids, pad], axis=1)
                attention_mask = np.array([encoding.attention_mask[:128]], dtype=np.int64)
                if attention_mask.shape[1] < 128:
                    pad = np.zeros((1, 128 - attention_mask.shape[1]), dtype=np.int64)
                    attention_mask = np.concatenate([attention_mask, pad], axis=1)
            else:
                encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')
                input_ids = encoded['input_ids'].astype(np.int64)
                attention_mask = encoded['attention_mask'].astype(np.int64)
            
            if use_rknn:
                outputs = encoder.inference(inputs=[input_ids, attention_mask])
                last_hidden = outputs[0]
            else:
                outputs = encoder.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
                last_hidden = outputs[0]
            
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_emb = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(attention_mask.astype(np.float32), axis=1, keepdims=True), a_min=1e-9, a_max=None)
            embedding = sum_emb / sum_mask
            
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            embedding = embedding / norms
            
            batch_embeddings.append(embedding[0])
        
        all_embeddings.extend(batch_embeddings)
        
        if i > 0 and i % 200 == 0:
            print(f"  进度: {i}/{n_samples} ({i*100//n_samples}%)", flush=True)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"\n编码完成, 形状: {embeddings.shape}")
    
    print("\n构建 FAISS 索引...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    output_path = f"{data_dir}/text2vec_768_full_vector.faiss"
    faiss.write_index(index, output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"索引保存: {output_path}")
    print(f"索引大小: {size_mb:.1f} MB")
    
    if use_rknn:
        encoder.release()
    
    print("\n完成!")
    return True


if __name__ == "__main__":
    data_dir = "/userdata/medical_rag_full"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    build_text2vec_index(data_dir)
