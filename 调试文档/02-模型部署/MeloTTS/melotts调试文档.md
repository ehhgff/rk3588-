# MeloTTS RK3588 部署问题总结

## 1. 问题概述

在 RK3588 开发板上部署 MeloTTS 模型时，遇到段错误（Segmentation fault）问题。经过详细调试，确定根本原因是 **RKNN API 版本与 NPU 驱动版本不兼容**。

### 1.1 环境信息

| 项目 | 版本 |
|------|------|
| 开发板 | RK3588 |
| 操作系统 | Debian GNU/Linux 11 (bullseye) |
| 内核版本 | 5.10.209 |
| RKNN API 版本 | 2.3.2 (2025-04-09) |
| NPU 驱动版本 | 0.9.6 |
| RKNN 模型版本 | 6 |

### 1.2 错误现象

```
rknn_api version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
rknn_driver version: 0.9.6
...
All inputs set, before rknn_inputs_set
rknn_ctx: 367351459072, n_input: 8
  inputs[0]: index=0, size=2048, type=8, fmt=3, buf=0x558890eff0
  ...
Segmentation fault
```

**关键发现：**
- 模型初始化成功
- 输入数据准备完成，所有参数检查无误
- 段错误发生在 `rknn_inputs_set` 调用时
- **RKNN API 2.3.2（2025年新版）与 NPU 驱动 0.9.6（旧版）不兼容**

---

## 2. 代码修改记录

### 2.1 必要的 Bug 修复（已保留）

以下修复是程序正确运行所必需的，**不应改回**：

#### 2.1.1 输入索引错误修复

**位置：** `melotts.cc` 第 305 行（`inference_decoder_model` 函数）

**问题：** decoder 模型中 `inputs[2].index` 被错误设置为 1，与 `inputs[1].index` 冲突。

**修复：**
```cpp
// 错误代码
inputs[2].index = 1;

// 正确代码（已保留）
inputs[2].index = 2;
```

**为什么应该是 2：**

根据 decoder 模型的输入定义（见 `melotts.cc` 注释）：
```cpp
// Set Input Data
// ["attn", "y_mask", "g", "m_p", "logs_p", "noise_scale"],
```

输入顺序和索引对应关系：
| 索引 | 输入名称 | 说明 |
|------|----------|------|
| 0 | `attn` | 注意力矩阵 |
| 1 | `y_mask` | 输出掩码 |
| 2 | `g` | 全局特征 |
| 3 | `m_p` | 均值参数 |
| 4 | `logs_p` | 对数标准差参数 |
| 5 | `noise_scale` | 噪声缩放因子 |

**关键原因：**
1. **RKNN 模型输入索引必须唯一** - 每个输入张量都有唯一的索引标识
2. **索引必须与模型定义一致** - 模型转换时定义的输入顺序决定了索引值
3. **`inputs[1].index` 已经是 1**（对应 `y_mask`），所以 `inputs[2]` 必须是 2（对应 `g`）
4. **错误的索引会导致** - `rknn_inputs_set` 无法正确匹配输入数据，可能引发段错误或错误结果

**代码验证：**
```cpp
inputs[0].index = 0;  // attn
inputs[1].index = 1;  // y_mask
inputs[2].index = 2;  // g（原来是 1，错误！与 y_mask 冲突）
inputs[3].index = 3;  // m_p
inputs[4].index = 4;  // logs_p
inputs[5].index = 5;  // noise_scale
```

#### 2.1.2 未初始化向量修复

**位置：** `process.cc` 第 137 行

**问题：** `attn_mask` 向量未初始化，导致未定义行为和潜在的内存错误。

**修复：**
```cpp
// 错误代码
std::vector<int> attn_mask(predicted_lengths_max * x_mask_size);

// 正确代码（已保留）
std::vector<int> attn_mask(predicted_lengths_max * x_mask_size, 0);
```

**说明：** 向量必须初始化为 0，否则可能包含垃圾值。

#### 2.1.3 缺少词表文件

**问题：** 缺少 `lexicon.txt` 和 `tokens.txt` 导致程序无法运行。

**解决：** 上传文件到 `/data/melotts/cpp/model/`

---

### 2.2 调试测试代码（已移除）

以下代码仅用于调试测试，**已移除**：

#### 2.2.1 SDK 版本查询

**位置：** `melotts.cc` 第 78-84 行

**已移除的代码：**
```cpp
// Get SDK version
rknn_sdk_version version;
ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
if (ret == RKNN_SUCC) {
    printf("rknn_api version: %s\n", version.api_version);
    printf("rknn_driver version: %s\n", version.drv_version);
}
```

#### 2.2.2 函数入口调试打印

**位置：** `melotts.cc`

**已移除的代码：**
```cpp
// inference_encoder_model 函数
printf("inference_encoder_model start\n");
printf("inputs/outputs memset done\n");
printf("sdp_ratio: %f, noise_scale_w: %f\n", sdp_ratio, noise_scale_w);

// inference_melotts_model 函数
printf("inference_melotts_model start\n");
printf("Vectors initialized\n");
printf("ja_bert initialized, size: %zu\n", ja_bert.size());
printf("Before inference_encoder_model\n");
printf("After inference_encoder_model, ret: %d\n", ret);
```

#### 2.2.3 输入参数调试打印

**位置：** `melotts.cc` 第 228-232 行

**已移除的代码：**
```cpp
printf("All inputs set, before rknn_inputs_set\n");
printf("rknn_ctx: %lu, n_input: %d\n", (unsigned long)app_ctx->rknn_ctx, n_input);
for (int i = 0; i < n_input; i++) {
    printf("  inputs[%d]: index=%u, size=%u, type=%d, fmt=%d, buf=%p\n", 
           i, inputs[i].index, inputs[i].size, inputs[i].type, inputs[i].fmt, inputs[i].buf);
}
```

#### 2.2.4 主流程调试打印

**位置：** `main.cc` 第 140-163 行

**已移除的代码：**
```cpp
printf("Before convert\n");
printf("After convert, phones_bef size: %zu, tones_bef size: %zu\n", phones_bef.size(), tones_bef.size());
printf("After intersperse, phones size: %zu\n", phones.size());
printf("After pad_or_trim, phones size: %zu\n", phones.size());
printf("Before inference_melotts_model\n");
printf("After inference_melotts_model, output_lengths: %d\n", output_lengths);
```

---

## 3. 待解决的核心问题

### 3.1 版本不兼容（根本原因）

| 组件 | 当前版本 | 要求版本 | 状态 |
|------|----------|----------|------|
| RKNN API | 2.3.2 | - | ✓ |
| NPU 驱动 | 0.9.6 | 0.9.8+ | ✗ |

**分析：**
- RKNN API 2.3.2 是 2025 年发布的新版本
- NPU 驱动 0.9.6 是旧版本
- API 期望的驱动功能与旧驱动不兼容，导致 `rknn_inputs_set` 内部访问无效内存

---

## 4. 解决方案

### 方案 1：更新 NPU 驱动（推荐）

将 NPU 驱动从 0.9.6 更新到 0.9.8 或更高版本。

**步骤：**
1. 获取开发板 SDK 内核源码
2. 下载 RKNPU 0.9.8 驱动源码
3. 修改驱动源码适配内核版本（如需要）
4. 编译内核和驱动
5. 安装新内核并重启

**参考教程：**
- https://www.cnblogs.com/sivon/p/19198733

**注意事项：**
- 当前内核为 5.10.209，教程主要针对 6.1 内核
- 可能需要适配 5.10 内核的修改
- 确保内核头文件和编译工具已安装

### 方案 2：使用兼容的 RKNN 库

尝试使用与驱动 0.9.6 兼容的旧版本 `librknnrt.so`。

**可用库文件：**
- `/home/ubuntu/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/3rdparty/rknpu2/Linux/aarch64/librknnrt.so`
- `/home/ubuntu/桌面/ai/MeloTTS/lubancat_ai_manual_code/dev_env/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so`

**操作：**
```bash
# 备份原库
adb shell "cp /lib/librknnrt.so /lib/librknnrt.so.backup"

# 上传新库
adb push ~/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/3rdparty/rknpu2/Linux/aarch64/librknnrt.so /lib/
```

### 方案 3：重新转换模型

使用与驱动 0.9.6 兼容的 RKNN-Toolkit2 版本重新转换模型。

**需要：**
- 找到与驱动 0.9.6 兼容的 RKNN-Toolkit2 版本（约 1.4.x 或 1.5.x）
- 原始 PyTorch 模型文件
- 重新执行模型转换流程

---

## 5. 调试代码修改

### 5.1 melotts.cc

**添加 SDK 版本查询：**
```cpp
rknn_sdk_version version;
ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
if (ret == RKNN_SUCC) {
    printf("rknn_api version: %s\n", version.api_version);
    printf("rknn_driver version: %s\n", version.drv_version);
}
```

**添加输入参数调试：**
```cpp
printf("rknn_ctx: %lu, n_input: %d\n", (unsigned long)app_ctx->rknn_ctx, n_input);
for (int i = 0; i < n_input; i++) {
    printf("  inputs[%d]: index=%u, size=%u, type=%d, fmt=%d, buf=%p\n", 
           i, inputs[i].index, inputs[i].size, inputs[i].type, inputs[i].fmt, inputs[i].buf);
}
```

### 5.2 process.cc

**初始化 attn_mask：**
```cpp
std::vector<int> attn_mask(predicted_lengths_max * x_mask_size, 0);
```

---

## 6. 验证步骤

### 6.1 检查 NPU 版本

```bash
adb shell "sudo cat /sys/kernel/debug/rknpu/version"
```

期望输出：
```
RKNPU driver: v0.9.8
```

### 6.2 运行测试

```bash
adb shell "cd /data/melotts/cpp && ./build/build_rk3588_linux/melotts_demo \
    --input_text '你好' --language ZH --output_filename test.wav"
```

---

## 7. 相关文件路径

| 类型 | 路径 |
|------|------|
| 主程序 | `/data/melotts/cpp/build/build_rk3588_linux/melotts_demo` |
| 编码器模型 | `/data/melotts/cpp/model/encoder-ZH_MIX_EN.rknn` |
| 解码器模型 | `/data/melotts/cpp/model/decoder-ZH_MIX_EN.rknn` |
| 词表文件 | `/data/melotts/cpp/model/lexicon.txt` |
| token 文件 | `/data/melotts/cpp/model/tokens.txt` |
| RKNN 库 | `/lib/librknnrt.so` |
| 源码目录 | `/home/ubuntu/桌面/ai/MeloTTS/lubancat_ai_manual_code/example/melotts/cpp/` |

---

## 8. 参考链接

- [RK3588 NPU 驱动更新教程](https://www.cnblogs.com/sivon/p/19198733)
- [MeloTTS 官方文档](https://github.com/myshell-ai/MeloTTS)
- [RKNN 用户手册](https://github.com/rockchip-linux/rknpu2)
