# InternVL3-1B 模型部署问题总结

## 一、模型文件说明

| 文件 | 用途 | 格式 |
|------|------|------|
| `internvl3-1b_vision_fp16_rk3588.rknn` | 视觉编码器 | RKNN |
| `internvl3-1b_w8a8_rk3588.rkllm` | 语言模型 | RKLLM (W8A8量化) |

## 二、部署过程中遇到的问题及解决方案

### 问题 1：GLIBC 版本不兼容

**现象：**
```
./llm_demo: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
./llm_demo: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
```

**原因：**
- PC端交叉编译器（Ubuntu 24.04）使用 GLIBC 2.39
- RK3588 开发板使用 GLIBC 2.31（Debian 11）
- 版本不匹配导致无法运行

**解决方案：**
给开发板联网并升级 GLIBC：
```bash
# 开发板联网
sudo ip addr add 192.168.100.2/24 dev eth0
sudo ip route add default via 192.168.100.1
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# 升级 libc6
sudo apt-get update
sudo apt-get install libc6
```

---

### 问题 2：开发板无法联网

**现象：**
无法使用 apt-get 安装软件包

**解决方案：**
通过网线共享电脑网络：
```bash
# 电脑端配置
sudo ip addr add 192.168.100.1/24 dev <usb网卡>
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o <主网卡> -j MASQUERADE

# 开发板端配置
sudo ip addr add 192.168.100.2/24 dev eth0
sudo ip route add default via 192.168.100.1
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

---

### 问题 3：缺少头文件和库文件

**现象：**
```
fatal error: rkllm.h: 没有那个文件或目录
```

**解决方案：**
将 RKLLM Runtime 的头文件和库复制到编译目录：
```bash
adb push rkllm.h /data/rkllm_src/src/
adb push librkllmrt.so /data/rkllm_src/
```

---

### 问题 4：CMakeLists.txt 路径问题

**现象：**
CMake 找不到源文件或库文件

**解决方案：**
修改 CMakeLists.txt 使用绝对路径或本地路径：
```cmake
include_directories(${CMAKE_SOURCE_DIR}/src)
set(RKLLM_RT_LIB ${CMAKE_SOURCE_DIR}/librkllmrt.so)
target_link_libraries(llm_demo ${RKLLM_RT_LIB})
```

---

### 问题 5：NPU 驱动版本警告

**现象：**
```
W rkllm: Warning: Your rknpu driver version is too low, please upgrade to 0.9.7
I rkllm: rknpu driver version: 0.9.6
```

**解决方案：**
1. 下载新驱动：`rknpu_driver_0.9.8_20241009.tar.bz2`
2. 在开发板上编译安装（需要内核头文件）
3. 或忽略警告继续使用（0.9.6 也能正常工作）

---

### 问题 6：Tokenizer 编码问题

**现象：**
```
terminate called after throwing an instance of 'std::invalid_argument'
what(): invalid character
```

**原因：**
输入中文字符时，tokenizer 无法编码

**解决方案：**
1. 使用英文输入
2. 修改代码添加 UTF-8 支持：
```cpp
#include <locale>
setlocale(LC_ALL, "zh_CN.UTF-8");
```
3. 使用支持中文的 tokenizer 配置

---

### 问题 7：输出中有大量 [PAD] 填充符

**现象：**
输出显示 `[PAD151935]` 等填充符

**原因：**
模型量化精度问题，原模型（DeepSeek-R1-Distill-Qwen-1.5B）量化后存在兼容性问题
问题描述：https://github.com/airockchip/rknn-llm/issues/424
**解决方案：**
更换模型为 `internvl3-1b_w8a8_rk3588.rkllm`，该模型量化精度正常，无填充符问题

---

## 三、成功部署步骤

### 1. 环境准备
```bash
# 开发板联网
sudo ip addr add 192.168.100.2/24 dev eth0
sudo ip route add default via 192.168.100.1
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# 安装编译工具
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

### 2. 上传源代码
```bash
adb push src/ /data/rkllm_src/
adb push CMakeLists.txt /data/rkllm_src/
adb push rkllm.h /data/rkllm_src/src/
adb push librkllmrt.so /data/rkllm_src/
```

### 3. 编译
```bash
adb shell
cd /data/rkllm_src
mkdir build && cd build
cmake ..
make
cp llm_demo /data/demo_Linux_aarch64/
```

### 4. 运行
```bash
cd /data/demo_Linux_aarch64
export LD_LIBRARY_PATH=./lib
./llm_demo ./internvl3-1b_w8a8_rk3588.rkllm 2048 4096
```

---

## 四、模型参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| max_new_tokens | 最大生成token数 | 2048 |
| max_context_len | 最大上下文长度 | 4096 |
| npu_core_num | NPU核心数 | 3 |

---

## 五、相关文件路径

| 文件 | 路径 |
|------|------|
| 模型文件 | `/home/ubuntu/桌面/ai/InternVL3-1B/` |
| RKLLM SDK | `/home/ubuntu/桌面/ai/deepseek/rknn-llm/` |
| 编译好的demo | `/data/demo_Linux_aarch64/` |
| 开发板源码 | `/data/rkllm_src/` |

---

## 六、参考资料

- RKLLM 官方文档：`/home/ubuntu/桌面/ai/deepseek/rknn-llm/README.md`
- API Demo：`/home/ubuntu/桌面/ai/deepseek/rknn-llm/examples/rkllm_api_demo/`
- 多模态Demo：`/home/ubuntu/桌面/ai/deepseek/rknn-llm/examples/multimodal_model_demo/`
- 模型下载：[rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8) (获取码: rkllm)
