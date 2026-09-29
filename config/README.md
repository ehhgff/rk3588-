# config - 配置文件

本目录包含各种配置文件，用于调整系统行为和性能参数。

## 文件说明

### config.yaml
**默认配置文件**
- 用途: 标准配置，平衡性能和资源使用
- 适用: 大多数场景
- 参数:
  - VAD阈值: 0.7
  - 连续语音帧数: 5
  - 静音检测时间: 0.7秒
  - TTS语速: 0.6
  - 音量: 30%

### config_optimized.yaml
**优化版配置文件**
- 用途: 针对RK3588优化的配置
- 适用: 生产环境，追求最佳性能
- 优化项:
  - 更快的响应速度
  - 更低的内存占用
  - NPU加速参数优化

### config_fast.yaml
**快速响应配置**
- 用途: 最小化延迟
- 适用: 对响应速度要求高的场景
- 特点:
  - 减少等待时间
  - 降低识别精度换取速度
  - 快速返回结果

### config_cot.yaml
**Chain-of-Thought配置**
- 用途: 支持思维链推理
- 适用: 需要详细解释的场景
- 特点:
  - LLM生成更详细的回答
  - 展示推理过程
  - 适合医疗咨询场景

## 配置参数说明

### VAD参数
```yaml
vad:
  threshold: 0.7              # VAD阈值 (0.0-1.0)
  speech_frames: 5            # 连续语音帧数触发
  silence_frames: 23          # 静音帧数停止 (0.7秒)
```

### 音频参数
```yaml
audio:
  sample_rate: 16000          # 采样率
  record_device: "plughw:0,0" # 录音设备
  play_device: "plughw:1,0"   # 播放设备
  volume: 30                  # 音量 (%)
```

### TTS参数
```yaml
tts:
  speed: 0.6                  # 语速 (0.5-2.0)
  output_format: "FLOAT_LE"   # 输出格式
  sample_rate: 44100          # 输出采样率
```

## 使用方法

1. **选择配置**: 根据场景选择合适的配置文件
2. **加载配置**: 在主程序中指定配置文件
3. **自定义**: 复制现有配置并修改参数

## 切换配置示例

```bash
# 使用优化配置
cp config_optimized.yaml config.yaml

# 或使用快速配置
cp config_fast.yaml config.yaml
```
