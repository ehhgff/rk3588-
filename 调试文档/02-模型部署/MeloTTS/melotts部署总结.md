# MeloTTS RK3588 部署完整总结

## 一、部署环境

| 项目 | 版本/信息 |
|------|----------|
| 开发板 | RK3588 |
| 操作系统 | Buildroot 2021.11（非 Debian） |
| NPU 驱动版本 | 0.9.8 |
| 交叉编译器 | gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu |
| RKNN Toolkit | 2.3.0 |

---

## 二、部署步骤

### 2.1 代码修改

#### 1. CMakeLists.txt 路径修正
**文件：** `melotts/cpp/CMakeLists.txt`

**修改内容：**
```cmake
# 原代码
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../utils/ utils.out)

# 修改为
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)
```

**原因：** 3rdparty 和 utils 目录在 example 目录下，不在 melotts 目录下

#### 2. 移除 filesystem 头文件
**文件：** `melotts/cpp/melotts.h`

**修改内容：**
```cpp
// 删除此行
#include <filesystem>
```

**原因：** GCC 7.5.0 不支持 C++17 的 filesystem 库

#### 3. 修正 libsndfile 路径
**文件：** `example/3rdparty/CMakeLists.txt`

**修改内容：**
```cmake
# 原代码（使用系统库）
set(LIBSNDFILE_INCLUDES /usr/include PARENT_SCOPE)
set(LIBSNDFILE /usr/lib/aarch64-linux-gnu/libsndfile.so PARENT_SCOPE)

# 修改为（使用 3rdparty 中的库）
set(LIBSNDFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libsndfile)
set(LIBSNDFILE_INCLUDES ${LIBSNDFILE_PATH}/include PARENT_SCOPE)
set(LIBSNDFILE ${LIBSNDFILE_PATH}/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libsndfile.a PARENT_SCOPE)
```

**原因：** 交叉编译时需要使用 ARM 版本的库，而不是 x86_64 系统库

#### 4. 输入索引错误修复（关键）
**文件：** `melotts/cc/melotts.cc` 第 305 行

**修改内容：**
```cpp
// 错误代码
inputs[2].index = 1;

// 正确代码
inputs[2].index = 2;
```

**原因：** decoder 模型输入索引必须唯一，inputs[1].index 已经是 1，inputs[2] 必须是 2

#### 5. 未初始化向量修复
**文件：** `melotts/cpp/process.cc` 第 137 行

**修改内容：**
```cpp
// 错误代码
std::vector<int> attn_mask(predicted_lengths_max * x_mask_size);

// 正确代码
std::vector<int> attn_mask(predicted_lengths_max * x_mask_size, 0);
```

**原因：** 向量必须初始化为 0，否则可能包含垃圾值

#### 6. 增大预测长度限制
**文件：** `melotts/cpp/process.h`

**修改内容：**
```cpp
// 原代码
#define PREDICTED_LENGTHS_MAX MAX_LENGTH*2

// 修改为
#define PREDICTED_LENGTHS_MAX MAX_LENGTH*4
```

**原因：** 长文本合成时，predicted_lengths_max_real 超过限制导致内容截断

---

### 2.2 编译步骤

```bash
# 1. 设置交叉编译器路径
export PATH="/home/ubuntu/桌面/ai/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH"

# 2. 进入代码目录
cd ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp

# 3. 执行编译
./build-linux.sh -t rk3588

# 4. 编译输出在
# - 可执行文件：build/build_rk3588_linux/melotts_demo
# - 安装目录：install/rk3588_linux/
```

---

### 2.3 部署到开发板

```bash
# 1. 创建部署目录
adb shell "mkdir -p /data/melotts_deploy"

# 2. 上传可执行文件
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp/install/rk3588_linux/melotts_demo /data/melotts_deploy/

# 3. 上传库文件
adb shell "mkdir -p /data/melotts_deploy/lib"
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp/install/rk3588_linux/lib/ /data/melotts_deploy/

# 4. 上传模型文件
adb shell "mkdir -p /data/melotts_deploy/model"
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp/encoder-ZH_MIX_EN.rknn /data/melotts_deploy/model/
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp/decoder-ZH_MIX_EN.rknn /data/melotts_deploy/model/

# 5. 上传词表文件
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/model/lexicon.txt /data/melotts_deploy/model/
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/model/tokens.txt /data/melotts_deploy/model/

# 6. 上传兼容的 librknnrt.so（关键）
adb push ~/桌面/ai/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /data/melotts_deploy/lib/
```

---

### 2.4 运行程序

```bash
# 基本用法（推荐参数：speak_id=1, speed=0.7）
adb shell "cd /data/melotts_deploy && LD_LIBRARY_PATH=./lib ./melotts_demo \
  --encoder_model_path ./model/encoder-ZH_MIX_EN.rknn \
  --decoder_model_path ./model/decoder-ZH_MIX_EN.rknn \
  --input_text '你好，欢迎使用MeloTTS' \
  --output_filename ./audio.wav \
  --speak_id 1 \
  --speed 0.7"

# 快速使用（使用默认推荐参数）
adb shell "cd /data/melotts_deploy && LD_LIBRARY_PATH=./lib ./melotts_demo \
  --input_text '你好，欢迎使用MeloTTS' \
  --output_filename ./audio.wav"

# 调整语速（范围 0.6-1.0，0.7 最佳）
adb shell "cd /data/melotts_deploy && LD_LIBRARY_PATH=./lib ./melotts_demo \
  --input_text '你好，欢迎使用MeloTTS' \
  --output_filename ./audio.wav \
  --speed 0.7"
```

---

## 三、遇到的问题及解决方案

### 3.1 开发板没有 apt-get

**现象：**
```
/bin/bash: line 1: apt-get: command not found
```

**原因：** 开发板使用的是 Buildroot 系统，不是 Debian/Ubuntu

**解决方案：** 在 PC 上交叉编译，然后上传到开发板运行

---

### 3.2 缺少 cmake 和 g++

**现象：**
```
cmake: command not found
g++: command not found
```

**原因：** Buildroot 系统是精简的嵌入式系统，没有开发工具

**解决方案：** 在 PC 上使用交叉编译工具链编译

---

### 3.3 段错误（Segmentation fault）

**现象：**
```
Segmentation fault
```

**原因：** librknnrt.so 版本与 NPU 驱动版本不兼容

**解决方案：**
```bash
# 使用与 NPU 驱动 0.9.8 兼容的 librknnrt.so
adb push ~/桌面/ai/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /data/melotts_deploy/lib/

# 运行时使用 LD_LIBRARY_PATH 指定库路径
LD_LIBRARY_PATH=./lib ./melotts_demo ...
```

---

### 3.4 文本过长被截断

**现象：**
```
predicted_lengths_max_real > PREDICTED_LENGTHS_MAX
```

**原因：** 长文本合成的预测长度超过了预设的最大值

**解决方案：**
```cpp
// 修改 process.h
#define PREDICTED_LENGTHS_MAX MAX_LENGTH*4  // 原来是 *2
```

---

### 3.5 语音速度太快

**解决方案：** 使用 `--speed` 参数调整语速
```bash
--speed 0.8  # 慢一点，更自然
--speed 0.9  # 稍微慢一点
--speed 1.0  # 默认速度
```

---

### 3.6 BERT 模型未启用

**现象：** 语音质量不够自然，韵律不够流畅

**原因：**
1. 代码中 BERT 部分未完成（只有 TODO 注释）
2. 缺少 bert.rknn 模型文件

**代码位置：** `main.cc` 第 125 行
```cpp
if(!args.disable_bert) {
    // TODO init_melotts_model
    goto out;
}
```

**解决方案：** 当前无法实现，需要：
1. 获取 bert.rknn 模型文件
2. 完成 BERT 推理代码
3. 使用 `--disable_bert false` 启用

---

## 四、关键参数说明

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `--input_text` | 输入文本 | - | - |
| `--output_filename` | 输出音频路径 | audio.wav | - |
| `--speed` | 语速 | 1.0 | **0.7** (最佳) |
| `--speak_id` | 说话人ID | 1 | **1** (唯一可用) |
| `--language` | 语言 | ZH | ZH_MIX_EN |
| `--disable_bert` | 禁用BERT | true | true（当前必须） |

**注意：**
- 经过测试，模型**只支持 speak_id=1**（单说话人模型），其他说话人 ID 无声
- 语速 **0.7** 效果最佳，听起来最自然
- 语速范围：0.6-1.0，低于 0.6 或高于 1.0 可能导致音质下降

---

## 五、性能指标

| 指标 | 数值 |
|------|------|
| 模型初始化时间 | ~200ms |
| Encoder 推理时间 | ~160ms |
| Decoder 推理时间 | ~1600ms |
| RTF（实时率） | 0.3-0.7 |
| 音频采样率 | 44100 Hz |
| 音频格式 | 单声道，IEEE Float |

---

## 六、文件清单

### 6.1 开发板上的文件结构
```
/data/melotts_deploy/
├── melotts_demo          # 可执行文件
├── lib/
│   ├── librknnrt.so      # RKNN 运行时库
│   └── librga.so         # RGA 图像处理库
└── model/
    ├── encoder-ZH_MIX_EN.rknn   # Encoder 模型
    ├── decoder-ZH_MIX_EN.rknn   # Decoder 模型
    ├── lexicon.txt              # 中文词表
    └── tokens.txt               # 音素映射表
```

### 6.2 PC 上的源码修改文件
```
~/桌面/ai/MeloTTS/lubancat_ai_manual_code/
├── example/melotts/cpp/CMakeLists.txt          # 路径修正
├── example/melotts/cpp/melotts.h               # 移除 filesystem
├── example/melotts/cpp/process.h               # 增大长度限制
├── example/3rdparty/CMakeLists.txt             # libsndfile 路径
└── example/3rdparty/toolchain.cmake            # 新增交叉编译配置
```

---

## 七、librknnrt.so 永久设置

### 7.1 替换系统默认库（推荐）

**步骤：**
```bash
# 1. 备份系统原有的库
adb shell "cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.backup"

# 2. 用兼容版本替换
adb push ~/桌面/ai/rknn-toolkit2-2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

# 3. 验证替换成功
adb shell "ls -la /usr/lib/librknnrt.so*"
# 输出：
# -rw-rw-rw- 1 root root 7259064 Nov 11  2024 /usr/lib/librknnrt.so  (新版本)
# -rw-r--r-- 1 root root 6863800 Jan  1 08:39 /usr/lib/librknnrt.so.backup (原版本备份)
```

**效果：**
- ✅ 不需要 `LD_LIBRARY_PATH` 环境变量
- ✅ 直接运行：`./melotts_demo --input_text '你好' --output_filename ./audio.wav`
- ✅ 系统所有程序都使用兼容版本

**恢复原版本：**
```bash
adb shell "cp /usr/lib/librknnrt.so.backup /usr/lib/librknnrt.so"
```

### 7.2 其他方案（备选）

**方案 2：设置环境变量到启动脚本**
```bash
adb shell "echo 'export LD_LIBRARY_PATH=/data/melotts_deploy/lib:\$LD_LIBRARY_PATH' >> /etc/profile"
```

**方案 3：创建启动脚本**
```bash
# 创建启动脚本
adb shell "cat > /data/melotts_deploy/run.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/data/melotts_deploy/lib
./melotts_demo "\$@"
EOF
"

# 添加执行权限
adb shell "chmod +x /data/melotts_deploy/run.sh"

# 使用
adb shell "cd /data/melotts_deploy && ./run.sh --input_text '你好' --output_filename ./audio.wav"
```

---

## 八、清理开发板文件

删除不需要的编译相关文件，只保留运行所需：

```bash
adb shell "cd /data/melotts_deploy && rm -rf 3rdparty build utils CMakeLists.txt main.cc melotts.cc melotts.h process.cc process.h parse_args.h split.hpp lexicon.hpp"
```

**清理后的文件结构：**
```
/data/melotts_deploy/
├── melotts_demo              # 可执行文件 (524KB)
├── lib/
│   ├── librknnrt.so         # RKNN 运行时库 (7.3MB)
│   └── librga.so            # RGA 图像处理库 (197KB)
├── model/
│   ├── encoder-ZH_MIX_EN.rknn   # Encoder 模型 (25.3MB)
│   ├── decoder-ZH_MIX_EN.rknn   # Decoder 模型 (129.8MB)
│   ├── lexicon.txt              # 中文词表 (6.9MB)
│   └── tokens.txt               # 音素映射表 (655B)
└── *.wav                      # 生成的音频文件
```

---

## 九、总结

MeloTTS 成功部署到 RK3588 开发板，主要难点：

1. **交叉编译环境配置** - Buildroot 系统缺少开发工具
2. **版本兼容性** - librknnrt.so 必须与 NPU 驱动版本匹配
3. **代码 Bug 修复** - 输入索引错误、未初始化向量等
4. **长文本处理** - 需要增大预测长度限制
5. **库文件管理** - 替换系统库实现永久设置

当前语音质量尚可，但受限于：
- 没有启用 BERT（代码未完成）
- 模型量化（W8A8）带来的音质损失
- 轻量级模型的固有限制
