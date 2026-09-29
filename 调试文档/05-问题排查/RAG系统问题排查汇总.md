# RAG医疗咨询系统问题排查汇总

## 一、问题分类统计

| 问题类别 | 数量 | 占比 | 严重程度 |
|----------|------|------|----------|
| 环境配置错误 | 2 | 11.8% | ⭐⭐ |
| 数据处理错误 | 3 | 17.6% | ⭐⭐⭐ |
| 模型推理错误 | 2 | 11.8% | ⭐⭐⭐⭐ |
| 部署配置错误 | 5 | 29.4% | ⭐⭐⭐⭐⭐ |
| 运行时错误 | 3 | 17.6% | ⭐⭐⭐ |
| 逻辑错误 | 2 | 11.8% | ⭐⭐⭐⭐ |
| **总计** | **17** | **100%** | |

---

## 二、详细问题列表

### 2.1 环境配置错误 (2个)

#### 问题1: ModuleNotFoundError: No module named 'numpy'
**错误代码**: `ModuleNotFoundError: No module named 'numpy'`
**现象**: 导入numpy失败
**原因分析**: 未激活正确的conda环境
**解决方案**: 
```bash
conda activate python3.12-tk2-2.3
pip install numpy
```
**影响程度**: ⭐⭐ (影响开发环境)

#### 问题2: huggingface-cli：未找到命令
**错误代码**: `bash: huggingface-cli: command not found`
**现象**: 命令行工具未安装
**原因分析**: 仅安装了huggingface_hub库，未安装命令行工具
**解决方案**: 使用Python API替代
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="shibing624/text2vec-base-chinese", filename="pytorch_model.bin")
```
**影响程度**: ⭐⭐ (影响模型下载)

---

### 2.2 数据处理错误 (3个)

#### 问题3: RuntimeError: Error: 'is_trained' failed
**错误代码**: `RuntimeError: Error: 'is_trained' failed`
**现象**: FAISS索引未训练
**原因分析**: PQ索引需要训练数据，但未调用train方法
**解决方案**: 
```python
# 添加训练步骤
index = faiss.IndexHNSWPQ(dimension, pq_m, hnsw_m)
index.train(vectors)  # 关键：先训练
index.add(vectors)
```
**影响程度**: ⭐⭐⭐ (导致索引构建失败)

#### 问题4: JSON加载错误
**错误代码**: `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
**现象**: 无法解析JSON文件
**原因分析**: 文件格式为JSONL而非JSON数组
**解决方案**: 
```python
# 改为JSONL格式读取
with open(file_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
```
**影响程度**: ⭐⭐⭐ (数据加载失败)

#### 问题5: 所有结果都是"男科"
**错误代码**: 无错误代码，逻辑错误
**现象**: 科室分布严重不平衡，所有查询都返回男科结果
**原因分析**: 分层采样逻辑错误，导致只加载了单一科室数据
**解决方案**: 
```python
# 实现正确的分层采样
for dept in departments:
    dept_samples = min(len(samples), samples_per_dept)
    selected.extend(random.sample(samples, dept_samples))
```
**影响程度**: ⭐⭐⭐ (影响检索准确率)

---

### 2.3 模型推理错误 (2个)

#### 问题6: Cannot send a request, as the client has been closed
**错误代码**: `RuntimeError: Cannot send a request, as the client has been closed`
**现象**: SBERT编码器无法联网
**原因分析**: RK3588无网络连接，无法下载模型
**解决方案**: 
```bash
# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 推送本地缓存模型
adb push ~/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese /root/.cache/huggingface/hub/
```
**影响程度**: ⭐⭐⭐⭐ (导致系统无法运行)

#### 问题7: No sentence-transformers model found
**错误代码**: `OSError: No sentence-transformers model found`
**现象**: 模型路径错误
**原因分析**: 缓存路径不匹配或模型文件损坏
**解决方案**: 
```python
# 使用本地路径加载
model_path = "/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/*/"
model = SentenceTransformer(model_path, device='cpu')
```
**影响程度**: ⭐⭐⭐⭐ (模型无法加载)

---

### 2.4 部署配置错误 (5个)

#### 问题8: TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'
**错误代码**: `TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'`
**现象**: ONNX量化参数错误
**原因分析**: API版本不兼容
**解决方案**: 移除不支持的参数
```python
# 删除 optimize_model=True
quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8)
```
**影响程度**: ⭐⭐⭐⭐⭐ (模型量化失败)

#### 问题9: Invalid quantized_dtype 'dynamic_fixed_point-i8'
**错误代码**: `ValueError: Invalid quantized_dtype 'dynamic_fixed_point-i8'`
**现象**: RKNN量化类型错误
**原因分析**: 不支持的量化类型
**解决方案**: 使用支持的类型
```python
rknn.config(quantized_dtype='w8a8')  # 改为w8a8
```
**影响程度**: ⭐⭐⭐⭐⭐ (RKNN转换失败)

#### 问题10: The len of mean_values ([0, 0, 0]) for input 0 is wrong
**错误代码**: `ValueError: The len of mean_values ([0, 0, 0]) for input 0 is wrong`
**现象**: RKNN输入维度不匹配
**原因分析**: 文本模型输入不是图像，不需要mean/std配置
**解决方案**: 移除mean/std配置
```python
# 删除 mean_values/std_values 参数
rknn.config()
```
**影响程度**: ⭐⭐⭐⭐⭐ (RKNN配置错误)

#### 问题11: Dataset file ./dataset.txt not found!
**错误代码**: `FileNotFoundError: Dataset file ./dataset.txt not found!`
**现象**: 校准数据缺失
**原因分析**: RKNN量化需要校准数据
**解决方案**: 
```bash
# 创建校准数据文件
echo -e "高血压吃什么药\\n感冒发烧怎么办\\n..." > dataset.txt
```
**影响程度**: ⭐⭐⭐⭐⭐ (量化失败)

#### 问题12: Unsupport file 高血压吃什么药!
**错误代码**: `RuntimeError: Unsupport file 高血压吃什么药!`
**现象**: RKNN不支持文本输入
**原因分析**: RKNN主要支持图像/CV模型，文本模型兼容性差
**解决方案**: 改用ONNX Runtime部署
```bash
pip install onnxruntime
# 使用ONNX模型推理
```
**影响程度**: ⭐⭐⭐⭐⭐ (部署策略错误)

---

### 2.5 运行时错误 (3个)

#### 问题13: Address already in use
**错误代码**: `OSError: [Errno 98] Address already in use`
**现象**: 端口被占用
**原因分析**: 服务未正常关闭
**解决方案**: 
```python
# 允许地址重用
class ReusableTCPServer(HTTPServer):
    allow_reuse_address = True

server = ReusableTCPServer(('0.0.0.0', 8081), Handler)
```
**影响程度**: ⭐⭐⭐ (服务启动失败)

#### 问题14: ModuleNotFoundError: No module named 'rank_bm25'
**错误代码**: `ModuleNotFoundError: No module named 'rank_bm25'`
**现象**: BM25库缺失
**原因分析**: 依赖未安装
**解决方案**: 
```bash
# 从开发机推送已安装的包
adb push /path/to/python/site-packages/rank_bm25 /userdata/medical_rag/
export PYTHONPATH=/userdata/medical_rag:$PYTHONPATH
```
**影响程度**: ⭐⭐⭐ (检索功能缺失)

#### 问题15: Destination Host Unreachable
**错误代码**: `socket.gaierror: [Errno -2] Name or service not known`
**现象**: RK3588无法联网
**原因分析**: 网络配置问题
**解决方案**: 
```bash
# 设置ADB反向代理
adb reverse tcp:8080 tcp:8080
export HTTP_PROXY=http://127.0.0.1:8080
```
**影响程度**: ⭐⭐⭐ (依赖下载失败)

---

### 2.6 逻辑错误 (2个)

#### 问题16: 高血压返回妇产科结果
**错误代码**: 无错误代码，逻辑错误
**现象**: "高血压吃什么药"返回"孕妇低血压吃什么好"
**原因分析**: 
1. 向量相似度被"血压"关键词主导
2. 未区分"高"vs"低"反义词
3. 未过滤特殊人群结果
**解决方案**: 
```python
# 1. 添加反义词检测
ANTONYM_PAIRS = [('高血压', '低血压'), ...]

# 2. 添加人群过滤
if not query_pop and result_pop:
    exclude_result()  # 排除特殊人群结果

# 3. 关键词增强
if synonym_match:
    score *= 1.5
```
**影响程度**: ⭐⭐⭐⭐ (影响用户体验)

#### 问题17: 月经不调未检测为妇产科
**错误代码**: 无错误代码，逻辑错误
**现象**: 妇科症状被错误分类
**原因分析**: 术语库不完整，缺少相关症状术语
**解决方案**: 
```python
# 扩展妇科术语
GYNECOLOGICAL_SYMPTOMS = {
    '月经异常': ['月经不调', '月经紊乱', '经期异常', ...],
}

# 更新科室映射
DEPT_MAPPING['妇产科'].extend(['月经异常', '白带异常'])
```
**影响程度**: ⭐⭐⭐⭐ (影响分类准确率)

---

## 三、问题解决策略总结

### 3.1 预防措施

| 问题类型 | 预防策略 |
|----------|----------|
| 环境配置 | 使用conda环境管理，创建requirements.txt |
| 数据处理 | 添加数据验证步骤，使用分层采样 |
| 模型推理 | 预下载模型，设置离线模式 |
| 部署配置 | 优先选择ONNX Runtime而非RKNN |
| 运行时 | 添加错误处理和重试机制 |
| 逻辑错误 | 完善术语库，添加过滤机制 |

### 3.2 调试技巧

1. **逐步验证**: 从数据加载→模型编码→检索→过滤，逐步验证
2. **日志输出**: 添加详细的日志输出，便于问题定位
3. **小数据测试**: 使用小数据集快速验证逻辑
4. **环境隔离**: 开发环境和部署环境保持一致

### 3.3 关键决策

1. **放弃RKNN**: 文本模型在RKNN上兼容性差，改用ONNX Runtime
2. **离线部署**: 预下载所有依赖，避免网络问题
3. **混合检索**: 结合向量和关键词检索，提升召回率
4. **过滤机制**: 添加反义词和人群过滤，提升准确率

---

## 四、性能优化成果

### 4.1 准确率提升
| 阶段 | 准确率 | 改进措施 |
|------|--------|----------|
| 初始版本 | 75.0% | MiniLM模型 |
| 优化后 | **87.5%** | SBERT+过滤机制 |

### 4.2 延迟优化
| 组件 | 优化前 | 优化后 |
|------|--------|--------|
| 编码延迟 | ~300ms | ~235ms |
| 检索延迟 | ~15ms | ~8ms |
| 总延迟 | ~315ms | **~245ms** |

### 4.3 内存占用
| 组件 | 内存占用 |
|------|----------|
| SBERT模型 | ~400MB |
| FAISS索引 | ~18MB |
| 数据文件 | ~15MB |
| **总计** | **~433MB** |

---

## 五、经验教训

### 5.1 技术选型经验
1. **模型选择**: SBERT在中文医疗数据上表现优于MiniLM
2. **部署方案**: ONNX Runtime比RKNN更适合文本模型
3. **检索策略**: 混合检索比单一检索效果好

### 5.2 开发流程经验
1. **环境管理**: 使用conda环境避免依赖冲突
2. **数据验证**: 添加数据质量检查步骤
3. **离线准备**: 预下载所有资源避免部署问题

### 5.3 问题排查经验
1. **日志记录**: 详细的日志是问题排查的关键
2. **逐步调试**: 从简单到复杂逐步验证
3. **版本控制**: 记录每次修改便于回滚

---

## 六、未来改进方向

### 6.1 技术改进
1. **模型优化**: 尝试更先进的编码器模型
2. **索引优化**: 优化FAISS参数提升检索速度
3. **缓存优化**: 添加查询结果缓存机制

### 6.2 功能扩展
1. **多语言支持**: 扩展支持英文医疗问答
2. **实时更新**: 支持在线更新医疗知识库
3. **个性化**: 基于用户历史优化推荐

### 6.3 运维改进
1. **监控告警**: 添加系统监控和性能告警
2. **自动化部署**: 实现一键部署和更新
3. **文档完善**: 持续完善技术文档和使用手册

---

*文档版本: v1.0*
*更新日期: 2026-03-28*
*问题数量: 17个*
*解决率: 100%*
