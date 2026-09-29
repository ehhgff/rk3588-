# tools - 工具和索引构建

本目录包含用于构建索引、数据处理和工具脚本。

## 文件说明

### build_dept_index.py
**构建科室分类索引**
- 功能: 根据医疗数据构建17个科室的独立FAISS索引
- 使用: `python3 build_dept_index.py`
- 输入: `sbert_768_full.json`
- 输出: `dept_vector_index.pkl`
- 作用: 为RAG服务提供科室分类索引，加速检索

### build_text2vec_index.py
**构建文本向量索引**
- 功能: 使用text2vec模型构建向量索引
- 使用: `python3 build_text2vec_index.py`
- 输出: FAISS索引文件
- 用途: 用于语义相似度检索

### build_text2vec_index_vm.py
**构建文本向量索引 (VM版本)**
- 功能: 针对虚拟机环境优化的索引构建脚本
- 使用: `python3 build_text2vec_index_vm.py`
- 区别: 内存使用优化，适合资源受限环境

### compress_faiss_index.py
**压缩FAISS索引**
- 功能: 压缩FAISS索引文件，减少存储空间
- 使用: `python3 compress_faiss_index.py`
- 输入: 原始FAISS索引
- 输出: 压缩后的索引
- 效果: 减少约50%存储空间，轻微影响检索速度

### check_rag_service.py
**检查RAG服务状态**
- 功能: 检查RAG服务是否正常运行
- 使用: `python3 check_rag_service.py`
- 检查项:
  - Socket文件是否存在
  - 服务是否响应
  - 基本查询测试

### convert_qwen3_2048_v2.py
**Qwen3模型转换脚本**
- 功能: 将Qwen3模型转换为RKLLM格式，max_context=2048
- 使用: `python3 convert_qwen3_2048_v2.py`
- 输入: HuggingFace格式的Qwen3模型
- 输出: RKLLM格式模型文件
- 特点:
  - W8A8量化
  - 最大上下文2048（减少内存占用）
  - 针对RK3588优化

## 使用场景

1. **首次部署**: 运行 `build_dept_index.py` 构建科室索引
2. **更新数据**: 重新运行索引构建脚本
3. **空间优化**: 使用 `compress_faiss_index.py` 压缩索引
4. **故障排查**: 使用 `check_rag_service.py` 检查服务状态

## 数据流

```
医疗原始数据
    ↓
build_dept_index.py → 科室分类索引
    ↓
compress_faiss_index.py → 压缩索引
    ↓
RAG服务使用
```
