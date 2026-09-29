# rk3588_deploy - RK3588部署文件

本目录包含专门针对RK3588开发板的部署文件和工具。

## 文件说明

### demo_queries_filtered.py
**演示查询过滤脚本**
- 功能: 过滤和筛选演示用的查询语句
- 用途: 准备测试用的医疗查询数据
- 使用: `python3 demo_queries_filtered.py`

### interactive_rag.py
**交互式RAG脚本**
- 功能: 提供交互式的RAG检索体验
- 用途: 命令行交互式测试RAG功能
- 使用: `python3 interactive_rag.py`

### medical_terminology_extended.py
**医疗术语扩展脚本**
- 功能: 扩展医疗术语库
- 用途: 增加更多医疗专业术语支持
- 使用: `python3 medical_terminology_extended.py`

### rk3588_server.py
**RK3588服务器脚本**
- 功能: RK3588专用的服务启动脚本
- 用途: 在RK3588上启动各项服务
- 使用: `python3 rk3588_server.py`

## RK3588特性

### NPU支持
- NPU型号: RKNN
- 算力: 6 TOPS
- 支持: INT8/FP16量化

### 硬件接口
- USB: 支持USB音频设备
- GPIO: 可连接按键、LED等
- UART: 串口调试

### 软件环境
- 操作系统: Linux (Ubuntu/Debian)
- Python: 3.10+
- RKNN Toolkit: 1.6+

## 部署步骤

1. **准备环境**
   ```bash
   # 安装依赖
   pip3 install -r requirements.txt
   ```

2. **部署模型**
   ```bash
   # 复制模型到指定目录
   cp -r models/ /data/voice_assistant/
   ```

3. **启动服务**
   ```bash
   # 使用RK3588优化配置启动
   python3 rk3588_server.py
   ```

## 性能优化

- 使用RKNN NPU加速推理
- 启用INT8量化减少内存占用
- 优化线程数以适应4核CPU

## 注意事项

1. 确保NPU驱动正确安装
2. 检查USB音频设备权限
3. 配置合适的Swap空间
4. 监控温度和功耗

## 相关文档

- [RK3588测试指南](../docs/医疗RAG语音助手测试指南.md)
- [部署检查清单](../docs/流式语音助手部署检查清单.md)
