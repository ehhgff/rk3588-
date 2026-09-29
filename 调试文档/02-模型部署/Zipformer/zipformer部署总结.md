# Zipformer 语音识别模型部署总结

## 一、模型简介

Zipformer 是一个**中英文语音识别 (ASR)** 模型，使用 k2-zipformer-streaming 架构，支持流式语音识别。

**模型来源：**
- HuggingFace: https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t

**模型特点：**
- 支持中英文双语识别
- 流式识别，低延迟
- 使用 NPU 加速推理
- RTF (实时率) < 1，实时性优秀

---

## 二、部署环境

### 2.1 硬件环境
- **开发板**：LubanCat 4 (RK3588)
- **NPU**：RK3588 内置 NPU (6 TOPS)
- **内存**：4GB LPDDR4
- **存储**：eMMC 32GB

### 2.2 软件环境
- **操作系统**：Debian 11 (Buildroot)
- **NPU 驱动版本**：0.9.8
- **RKNN Toolkit**：2.3.0

### 2.3 模型文件
| 文件 | 大小 | 用途 |
|------|------|------|
| `encoder-epoch-99-avg-1.rknn` | 113.9 MB | 编码器（特征提取） |
| `decoder-epoch-99-avg-1.rknn` | 7.7 MB | 解码器（生成文本） |
| `joiner-epoch-99-avg-1.rknn` | 6.2 MB | Joiner（对齐编码器和解码器） |
| `vocab.txt` | 56 KB | 词表文件 |
| `test.wav` | 180 KB | 测试音频 |

---

## 三、部署步骤

### 3.1 源码位置

```
~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/
├── cpp/
│   ├── main.cc                    # 主程序入口
│   ├── process.cc / process.h     # 处理逻辑
│   ├── zipformer.h                # 头文件
│   ├── rknpu2/zipformer.cc        # RKNN 实现
│   └── CMakeLists.txt             # 编译配置
├── python/
│   └── zipformer.py               # Python 版本
├── model/                         # 模型文件目录
└── README.md                      # 说明文档
```

### 3.2 编译（已预编译）

编译好的程序位置：
```
~/桌面/ai/rknn_model_zoo-2.3.0/build/build_rknn_zipformer_demo_rk3588_linux_aarch64_Release/rknn_zipformer_demo
```

如需重新编译：
```bash
cd ~/桌面/ai/rknn_model_zoo-2.3.0
./build-linux.sh -t rk3588 -a rknn_zipformer_demo -b Release
```

### 3.3 创建部署目录并上传文件

```bash
# 在开发板上创建目录
adb shell "mkdir -p /data/zipformer/model /data/zipformer/lib"

# 上传程序
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/build/build_rknn_zipformer_demo_rk3588_linux_aarch64_Release/rknn_zipformer_demo /data/zipformer/

# 上传模型文件
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/model/encoder-epoch-99-avg-1.rknn /data/zipformer/model/
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/model/decoder-epoch-99-avg-1.rknn /data/zipformer/model/
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/model/joiner-epoch-99-avg-1.rknn /data/zipformer/model/

# 上传词表和测试音频
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/model/vocab.txt /data/zipformer/model/
adb push ~/桌面/ai/rknn_model_zoo-2.3.0/examples/zipformer/model/test.wav /data/zipformer/model/
```

---

## 四、使用方法

### 4.1 运行语音识别

```bash
cd /data/zipformer
chmod +x rknn_zipformer_demo

./rknn_zipformer_demo \
    ./model/encoder-epoch-99-avg-1.rknn \
    ./model/decoder-epoch-99-avg-1.rknn \
    ./model/joiner-epoch-99-avg-1.rknn \
    ./model/test.wav
```

### 4.2 参数说明

| 参数 | 说明 |
|------|------|
| `encoder-epoch-99-avg-1.rknn` | 编码器模型，提取音频特征 |
| `decoder-epoch-99-avg-1.rknn` | 解码器模型，生成识别文本 |
| `joiner-epoch-99-avg-1.rknn` | Joiner 模型，对齐编码器和解码器 |
| `test.wav` | 输入音频文件（WAV 格式，16kHz，单声道） |

### 4.3 识别自己的音频

1. 准备音频文件（要求：WAV 格式，16kHz 采样率，单声道，16bit）
2. 上传到开发板：
   ```bash
   adb push your_audio.wav /data/zipformer/model/
   ```
3. 运行识别：
   ```bash
   cd /data/zipformer
   ./rknn_zipformer_demo \
       ./model/encoder-epoch-99-avg-1.rknn \
       ./model/decoder-epoch-99-avg-1.rknn \
       ./model/joiner-epoch-99-avg-1.rknn \
       ./model/your_audio.wav
   ```

---

## 五、测试结果

### 5.1 测试输出示例

```
-- init_zipformer_encoder_model use: 163.916000 ms
-- init_zipformer_decoder_model use: 8.394000 ms
-- init_zipformer_joiner_model use: 6.597000 ms
-- inference_zipformer_model use: 955.721008 ms

Real Time Factor (RTF): 0.956 / 5.841 = 0.164

Timestamp (s): 0.00, 0.48, 0.72, 0.88, 1.16, 1.40, 2.00, 2.04, 2.20, 2.36, 2.52, 2.68, 2.80, 3.36, 3.48, 3.64, 3.76, 3.88, 3.96, 4.04, 4.16, 4.28, 4.44, 4.60, 4.68, 5.16

Zipformer output: 对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢
```

### 5.2 性能指标

| 指标 | 数值 |
|------|------|
| 编码器初始化 | 163.92 ms |
| 解码器初始化 | 8.39 ms |
| Joiner 初始化 | 6.60 ms |
| 推理时间 | 955.72 ms |
| 音频长度 | 5.841 秒 |
| **RTF (实时率)** | **0.164** |

**RTF 说明：**
- RTF < 1 表示实时性良好
- RTF = 0.164 表示处理 1 秒音频只需要 0.164 秒
- 可以支持实时语音识别

---

## 六、文件清单

### 6.1 开发板上的文件结构
```
/data/zipformer/
├── rknn_zipformer_demo               # 可执行文件 (552KB)
├── model/
│   ├── encoder-epoch-99-avg-1.rknn  # 编码器模型 (113.9MB)
│   ├── decoder-epoch-99-avg-1.rknn  # 解码器模型 (7.7MB)
│   ├── joiner-epoch-99-avg-1.rknn   # Joiner 模型 (6.2MB)
│   ├── vocab.txt                    # 词表文件 (56KB)
│   └── test.wav                     # 测试音频 (180KB)
└── lib/                             # 库文件目录（可选）
```

### 6.2 总存储占用
- 程序：552 KB
- 模型文件：127.8 MB
- 词表和测试：236 KB
- **总计：约 128 MB**

---

## 七、与其他模型对比

| 模型 | 用途 | 大小 | 特点 |
|------|------|------|------|
| **MeloTTS** | 语音合成 | ~160 MB | 文本转语音 |
| **InternVL3-1B** | 多模态对话 | ~1.4 GB | 图文理解 |
| **Zipformer** | 语音识别 | ~128 MB | 语音转文字 |

**组合使用场景：**
1. **语音输入** → Zipformer（语音识别）→ 文本
2. **文本处理** → InternVL3-1B（对话理解）→ 回复文本
3. **语音输出** → MeloTTS（语音合成）→ 语音

形成完整的语音交互系统！

---

## 八、总结

Zipformer 语音识别模型成功部署到 RK3588 开发板：

### 已完成：
1. ✅ 模型文件上传到开发板
2. ✅ 程序编译和部署
3. ✅ 语音识别功能正常工作
4. ✅ 实时率 RTF = 0.164，性能优秀

### 模型能力：
- ✅ 中文语音识别：准确率高
- ✅ 英文语音识别：支持双语
- ✅ 实时识别：RTF < 1，支持流式处理
- ✅ NPU 加速：充分利用 RK3588 NPU

### 使用建议：
- 音频格式：WAV，16kHz，单声道，16bit
- 适用于：语音助手、语音输入、实时字幕等场景
- 可与 MeloTTS 和 InternVL3-1B 组合，构建完整语音交互系统
