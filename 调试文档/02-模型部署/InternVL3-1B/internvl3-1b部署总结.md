# InternVL3-1B 模型部署总结

## 一、部署环境

### 1.1 硬件环境
- **开发板**：LubanCat 4 (RK3588)
- **NPU**：RK3588 内置 NPU (6 TOPS)
- **内存**：4GB LPDDR4
- **存储**：eMMC 32GB

### 1.2 软件环境
- **操作系统**：Debian 11 (Buildroot)
- **NPU 驱动版本**：0.9.8
- **RKLLM Runtime**：1.2.3
- **RKNN Toolkit**：2.3.0

### 1.3 模型文件
| 文件 | 大小 | 用途 |
|------|------|------|
| `internvl3-1b_w8a8_rk3588.rkllm` | 761 MB | 语言模型 (W8A8量化) |
| `internvl3-1b_vision_fp16_rk3588.rknn` | 619 MB | 视觉编码器 (FP16) |

---

## 二、部署步骤

### 2.1 查找模型文件和部署代码

模型文件位置：
```
~/桌面/ai/internvl3-1b_w8a8_rk3588.rkllm
~/桌面/ai/internvl3-1b_vision_fp16_rk3588.rknn
```

部署代码位置：
```
~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/
```

### 2.2 创建部署目录并上传文件

```bash
# 在开发板上创建目录
adb shell "mkdir -p /data/internvl3"

# 上传程序
adb push ~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/install/demo_Linux_aarch64/demo /data/internvl3/
adb push ~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/install/demo_Linux_aarch64/imgenc /data/internvl3/

# 上传库文件
adb shell "mkdir -p /data/internvl3/lib"
adb push ~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/install/demo_Linux_aarch64/lib/ /data/internvl3/

# 上传模型文件
adb push ~/桌面/ai/internvl3-1b_w8a8_rk3588.rkllm /data/internvl3/
adb push ~/桌面/ai/internvl3-1b_vision_fp16_rk3588.rk3588.rknn /data/internvl3/

# 上传测试图片
adb push ~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/install/demo_Linux_aarch64/demo.jpg /data/internvl3/
```

### 2.3 解决依赖问题

**问题**：缺少 `libgomp.so.1` (OpenMP 库)

**解决**：从交叉编译器复制到开发板
```bash
# 查找并上传 libgomp
find /usr -name 'libgomp.so*' 2>/dev/null
adb push /usr/aarch64-linux-gnu/lib/libgomp.so.1 /data/internvl3/lib/
```

### 2.4 永久设置库文件

将库文件复制到系统目录，避免每次设置 LD_LIBRARY_PATH：

```bash
# 备份并替换 librkllmrt.so
adb shell "cp /usr/lib/librkllmrt.so /usr/lib/librkllmrt.so.backup 2>/dev/null"
adb shell "cp /data/internvl3/lib/librkllmrt.so /usr/lib/"

# 复制 libgomp.so.1
adb shell "cp /data/internvl3/lib/libgomp.so.1 /usr/lib/"

# 验证
adb shell "ls -la /usr/lib/librkllmrt.so* /usr/lib/libgomp.so*"
```

---

## 三、使用方法

### 3.1 完整版 (demo) - 支持图像输入

```bash
cd /data/internvl3
./demo demo.jpg ./internvl3-1b_vision_fp16_rk3588.rknn ./internvl3-1b_w8a8_rk3588.rkllm 256 2048 1
```

**参数说明：**
- `demo.jpg` - 输入图片路径
- `./internvl3-1b_vision_fp16_rk3588.rknn` - 视觉编码器模型
- `./internvl3-1b_w8a8_rk3588.rkllm` - 语言模型
- `256` - 最大生成 token 数
- `2048` - 最大上下文长度
- `1` - 使用的 NPU 核心数

**启动时间：** ~5.1 秒（包含视觉编码器加载和图像推理）

### 3.2 纯文本版 (demo_text) - 仅文本对话

创建纯文本版本，去掉视觉编码器加载：

**修改内容：**
1. 创建 `main_text.cpp` - 移除所有图像相关代码
2. 修改 `CMakeLists.txt` - 添加 `demo_text` 编译目标
3. 重新编译并部署

**使用方法：**
```bash
cd /data/internvl3
./demo_text ./internvl3-1b_w8a8_rk3588.rkllm 256 2048
```

**参数说明：**
- `./internvl3-1b_w8a8_rk3588.rkllm` - 语言模型路径
- `256` - 最大生成 token 数
- `2048` - 最大上下文长度

**启动时间：** ~1.0 秒（**节省 80% 时间**）

---

## 四、遇到的问题及解决方案

### 4.1 缺少 libgomp.so.1

**现象：**
```
./demo: error while loading shared libraries: libgomp.so.1: cannot open shared object file: No such file or directory
```

**解决：**
```bash
adb push /usr/aarch64-linux-gnu/lib/libgomp.so.1 /data/internvl3/lib/
adb shell "cp /data/internvl3/lib/libgomp.so.1 /usr/lib/"
```

### 4.2 视觉功能无法正常工作

**现象：**
- 模型可以加载和运行
- 纯文本对话正常
- 但无法正确识别图片内容

**可能原因：**
1. 图片 token 格式不匹配（`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`）
2. 视觉编码器模型与语言模型版本不匹配
3. RKLLM Runtime 版本与模型转换版本不一致

**解决：**
由于视觉功能调试复杂，且模型作为**纯文本对话模型**表现良好，建议：
- 日常使用 `demo_text` 纯文本版本
- 如需视觉功能，需要进一步调试或更换模型

---

## 五、性能对比

| 版本 | 加载时间 | 特点 | 推荐使用场景 |
|------|---------|------|-------------|
| **demo (完整版)** | ~5.1 秒 | 包含视觉编码器，理论上支持图像 | 需要图像理解功能时 |
| **demo_text (纯文本版)** | ~1.0 秒 | 仅语言模型，纯文本对话 | 日常文本对话 |

**时间节省：**
- 视觉编码器加载：627.14 ms → 0 ms
- 图像推理时间：3525.35 ms → 0 ms
- 总启动时间节省：**~4.1 秒 (80%)**

---

## 六、文件清单

### 6.1 开发板上的文件结构
```
/data/internvl3/
├── demo                              # 完整版程序 (7MB)
├── demo_text                         # 纯文本版程序 (72KB)
├── imgenc                            # 图像编码器程序 (7MB)
├── lib/
│   ├── librkllmrt.so                # RKLLM 运行时库 (7.5MB)
│   ├── librknnrt.so                 # RKNN 运行时库 (7.7MB)
│   └── libgomp.so.1                 # OpenMP 库 (452KB)
├── internvl3-1b_vision_fp16_rk3588.rknn   # 视觉编码器模型 (619MB)
├── internvl3-1b_w8a8_rk3588.rkllm         # 语言模型 (761MB)
└── demo.jpg                         # 测试图片 (245KB)
```

### 6.2 PC 上的源码修改文件
```
~/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/deploy/
├── src/main_text.cpp                 # 新增纯文本版本源码
└── CMakeLists.txt                    # 添加 demo_text 编译目标
```

---

## 七、系统库文件状态

```
/usr/lib/
├── librknnrt.so          # RKNN 运行时 (已替换为兼容版本，7.3MB)
├── librknnrt.so.backup   # 原版本备份 (6.9MB)
├── librkllmrt.so         # RKLLM 运行时 (新添加，7.5MB)
└── libgomp.so.1          # OpenMP (新添加，452KB)
```

---

## 八、总结

InternVL3-1B 模型成功部署到 RK3588 开发板：

### 已完成：
1. ✅ 模型文件上传到开发板
2. ✅ 依赖库问题解决 (libgomp.so.1)
3. ✅ 库文件永久设置（不需要 LD_LIBRARY_PATH）
4. ✅ 纯文本版本创建（启动速度提升 80%）
5. ✅ 文本对话功能正常工作

### 存在问题：
- ❌ 视觉理解功能无法正常工作（图片 token 格式或模型匹配问题）

### 建议：
- **日常使用**：使用 `demo_text` 纯文本版本，启动快，对话质量良好
- **视觉功能**：如需图像理解，建议尝试其他模型或进一步调试

### 模型能力：
- ✅ 中文对话：表现良好
- ✅ 英文对话：表现良好
- ✅ 文本生成：逻辑清晰
- ❌ 图像理解：当前不可用
