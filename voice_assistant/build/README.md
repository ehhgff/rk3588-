# build - 构建输出

本目录用于存放构建过程中生成的中间文件和输出文件。

## 当前文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `llm_client` | 280KB | LLM客户端可执行程序（C/C++编译） |
| `model_service_daemon` | 73KB | 模型服务守护进程（C/C++编译） |
| `Qwen3-0.6B_W8A8_RK3588_2048.rkllm` | ~800MB | Qwen3-0.6B RKLLM模型（W8A8量化，2048上下文） |
| `encoder-ZH_MIX_EN.onnx` | - | MeloTTS编码器ONNX模型（中英混合） |
| `encoder-ZH_MIX_EN.rknn` | - | MeloTTS编码器RKNN模型（中英混合，NPU加速） |
| `decoder-ZH_MIX_EN.onnx` | - | MeloTTS解码器ONNX模型（中英混合） |
| `decoder-ZH_MIX_EN.rknn` | - | MeloTTS解码器RKNN模型（中英混合，NPU加速） |

## 目录用途

### 编译生成的可执行文件
- 存放从C/C++源码编译生成的二进制文件
- 与源码分离，保持源码目录整洁

### ONNX模型文件
- 存放从PyTorch转换而来的ONNX模型
- 用于进一步转换为RKNN格式

### RKNN模型文件
- 存放转换后的RKNN模型
- 用于在RK3588 NPU上推理

### 优化后的模型
- 存放经过优化的模型文件
- 包括量化、剪枝等优化后的版本

## 文件命名规范

```
[模型名称]_[版本]_[优化类型].[格式]

示例:
- encoder_epoch99_avg1.onnx
- encoder_epoch99_avg1.rknn
- encoder_epoch99_avg1_fp16.rknn
```

## 注意事项

1. 本目录下的文件通常较大，不建议提交到Git
2. 模型文件可以通过脚本重新生成
3. 编译后的可执行文件如需重新生成，需要重新编译C/C++源码
4. 定期清理旧版本模型以节省空间
