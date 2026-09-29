# Qwen3-1.7B RK3588 部署指南

## 1. 模型转换完成 ✓

- **模型文件**: `Qwen3-1.7B_W8A8_RK3588.rkllm`
- **文件大小**: 2.3 GB
- **量化方式**: W8A8
- **目标平台**: RK3588

## 2. 开发板部署状态 ✓

已上传到开发板的文件：
```
/data/qwen3/
├── Qwen3-1.7B_W8A8_RK3588.rkllm  (模型文件)
└── librkllmrt.so                  (Runtime库)

/userdata/voice_assistant/
├── qwen3_llm_service.py           (服务程序)
├── qwen3_client.py                (客户端)
├── start_qwen3_service.sh         (启动脚本)
└── start_qwen3_simple.sh          (简化启动脚本)
```

## 3. 运行模型

### 3.1 使用 rknn-llm 示例程序

需要在开发板上编译 rknn-llm 的 C++ 示例：

```bash
# 进入开发板
adb shell

# 安装编译工具
sudo apt update
sudo apt install -y gcc g++ cmake

# 创建测试目录
mkdir -p /userdata/qwen3_test
cd /userdata/qwen3_test

# 复制必要的文件
cp /data/qwen3/librkllmrt.so .
cp /data/qwen3/Qwen3-1.7B_W8A8_RK3588.rkllm .

# 设置环境变量
export LD_LIBRARY_PATH=/userdata/qwen3_test:$LD_LIBRARY_PATH

# 运行测试 (需要编译 llm_demo)
```

### 3.2 使用 Python API

如果需要 Python 支持，需要：
1. 安装 rkllm-runtime Python 包
2. 或使用 ctypes 调用 librkllmrt.so

## 4. 下一步

由于 rknn-llm 的 Python API 需要额外的 runtime 包，建议：

1. **方案A**: 使用 C++ 编译 llm_demo 运行
2. **方案B**: 从 rknn-llm 官网下载完整的 runtime 包
3. **方案C**: 使用现有的 InternVL3 demo 程序测试

## 5. 参考文档

- [RKNN-LLM GitHub](https://github.com/airockchip/rknn-llm)
- [野火 Qwen3 部署教程](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/qwen3.html)
