# 更新记录 4 — ASR 管道缓冲彻底修复（2026-05-12）

## 问题概述

语音助手的 ASR（语音识别）从第一句话说出口到拿到识别文字，延迟高达 10~58 秒。根因不是模型推理慢，而是 **C 程序（sherpa-onnx-alsa）的 stderr 输出在管道模式下被 glibc 全缓冲（4KB/8KB）了**——识别结果一行只有几十字节，要攒满 4KB 才发送到 Python 端。

---

## 解决历程（含失败的尝试）

### 第 1 轮：stdout 管道阻塞修复

**问题**：`subprocess.Popen(stdout=subprocess.PIPE)` 但 stdout 从未被读取，64KB 管道缓冲区满后 ASR 进程阻塞。

**解决**：改用 `stdout=subprocess.DEVNULL`，废弃不需要的 stdout 管道。

**工具**：Python `subprocess.DEVNULL`

---

### 第 2 轮：stdbuf 行缓冲尝试

**问题**：C 库在 PIPE 模式下用全缓冲。

**尝试**：在命令前加 `stdbuf -eL`，通过 `LD_PRELOAD` 注入 `setvbuf(stderr, NULL, _IOLBF, 0)` 强制行缓冲。

**结果**：Buildroot 上的 BusyBox 对 `stdbuf` 支持不完整，实际未生效。

**工具**：`stdbuf`（GNU coreutils）、`LD_PRELOAD`

---

### 第 3 轮：select.select 轮询

**问题**：试图用 `select.select` 检测 pipe 是否有新数据。

**结果**：Buildroot 的 Python 中 `select.select` 对 pipe 不工作。

**解决**：改为纯轮询 + 空闲超时判断。

**工具**：Python `select.select`（失败）、time 轮询（成功）

---

### 第 4 轮：PTY（伪终端）替代 PIPE

**问题**：C 程序在 stderr 是 PIPE 时用全缓冲，在 stderr 是 **终端** 时自动用行缓冲。

**解决**：用 `pty.openpty()` 创建伪终端对，把 slave_fd 作为子进程的 stderr，master_fd 用 Python 读取。C 程序以为自己在写终端 → 每行立即刷新。

```
+--------------------+        +------------------+
| sherpa-onnx-alsa   |───────→|  pty master_fd   |───────→ Python reader
| (写 stderr)         | PTY   |  (行缓冲)         |         (实时收到)
+--------------------+        +------------------+
```

**工具**：Python `pty.openpty()`、`os.close()`、`open(fd, 'r', buffering=1)`

---

### 第 5 轮：PTY 引入的 Bug 修复

PTY 虽然解决了缓冲问题，但引入了一系列新 Bug：

| 问题 | 根因 | 解决方案 | 工具 |
|------|------|---------|------|
| `TypeError: bufsize` 崩溃 | `os.fdopen()` 不支持 `bufsize=` 参数关键词 | Python 3.10 API 差异 | `open(fd, 'r', buffering=1)` |
| `OSError [Errno 5]` 读线程崩溃 | PTY master 在子进程关闭后返回 I/O 错误 | 读线程 try/except OSError | Python `try/except` |
| 输出带 `]` 乱码、`\r\n` 换行 | 终端线路规程（line discipline）把 `\n` → `\r\n` | `tty.setraw(slave_fd)` 禁用终端处理 | Python `tty.setraw()` |
| 等不到结果就超时退出 | 读线程缩进错误，行处理代码在 `except` 外 | 重写 reader，行处理在 `try` 内 | 代码重构 |

---

### 第 6 轮：300ms 空闲检测算法

**问题**：检测到第一个字就返回，没有等后续补充（"你" vs "你好"）。

**解决**：基于 `len(_reader_lines)` 计数值稳定 300ms 不变才返回，不再用时间戳差。

```
_reader_lines 每收到一行 +1
if cur_count == prev_count:  # 300ms 没新行
    => 语音已经说完
```

**工具**：`_reader_lines` 列表、`_prev_lines_count`、`_lines_stable_since`

---

### 第 7 轮：日志清理

**问题**：`[RESULT] 识别:` 输出被终端控制序列覆盖，看不到。

**解决**：所有 `print()` 加 `flush=True`，去掉 DEBUG 杂音。

**工具**：Python `print(flush=True)`

---

### 第 8 轮：speech_end 计时修复

**问题**：`speech_end` 时间标记在 ASR 开始前就打点，"说话结束" → "播放首句" 的耗时包含了等待语音的时间。

**解决**：移到 ASR 返回有效识别文字之后，`speech_end` 从识别完成开始算。

**工具**：代码逻辑调整

---

### 第 9 轮：PTY 分段导致识别文字截断

**问题**：ASR 返回的文字被截断，"感冒了怎么办" 变成 "感冒了怎" 甚至只剩 "感冒了"。频繁出现。

**根因**：sherpa-onnx-alsa 用多个 `fprintf(stderr, ...)` 调用分段发送一行识别结果。PTY 工作在 raw 模式（`tty.setraw`）下，`/r` 和 `\n` 可能跟在下一个 chunk 里才到，和文字不在同一 chunk：

```
fprintf(stderr, "\r%d:", segment_id);     // chunk A: "0:"
fprintf(stderr, "%s", text);               // chunk B: "感冒了怎么办"
fprintf(stderr, "\n");                     // chunk C: "\n"
```

原来用 `open(master_fd, 'r', buffering=1)` 的 `readline()` — 等不到 delimiter 就不算一行，导致读到 chunk B 时因为没有尾部分隔符而卡在缓冲区里。

**解决**：废除基于文件对象的读取，改为 `os.read(master_fd, 4096)` 裸读 + 手动分割缓冲：

1. **`os.read()` 裸读**：每次读到原始字节，避免文件对象缓冲和行缓存
2. **双分隔符分割**：同时处理 `\r`（中间更新）和 `\n`（最终结果）
3. **buf 正则提取**：从剩余不完整的 buf 中用 `re.search(r'(\d+):([^\r\n]*)', buf)` 提取最新文本，即使缺尾部分隔符也能拿到文字
4. **`_extract_text()` 统一入口**：无论来自完整行还是 buf 提取，都用同一个函数处理

```
完整行分割（\n/\r）→ _extract_text(line, is_final)
剩余 buf    → re.search(r'(\d+):([^\r\n]*)', buf) → _extract_text(raw_text, False)
```

**工具**：Python `os.read()`、`re.search()`、手动 `\n`/`\r` 分割

---

### 第 10 轮：数据空闲超时替代 \n 触发

**问题**：依赖 `\n`（endpoint final frame）退出等待导致两大问题：
- ASR endpoint 触发需要停满 1.0s（rule2-min-trailing-silence），用户说话不停顿时 `\n` 要等下一句才来（20s+）
- 第 9 轮修复前依赖文字长度稳定（`len(_all_texts)` 不变 300ms），但 os.read 模式下 buf 提取会高频更新 `_all_texts`，长度法已不管用

**解决**：从基于长度的稳定检测 → 基于数据到达时间的空闲超时：

```
_last_data_time = [0.0]    # 每次 os.read 有数据就更新
退出条件 = _have_final (\n 到达) OR (有 >=2 字 && 空闲 >=800ms)
```

- `_last_data_time[0] = time.time()` 放在 reader 线程每次 `os.read` 成功之后
- 主循环检查 `time.time() - _last_data_time[0] >= 0.8`（800ms）
- 不再需要 `_prev_lines_count`、`_lines_stable_since`、`_has_text` 等复杂状态
- 800ms 空闲能可靠判断用户已停顿时语音结束，即使 ASR endpoint 还没触发

**工具**：`_last_data_time` 共享变量、reader 线程打时间戳

---

## 最终架构

```
麦克风 → [sherpa-onnx-alsa] ─PTY─→ os.read(master_fd, 4096) ─→ streaming_text ─→ RAG/LLM/TTS
         rule2=1.0
         enable_thinking=False
         800ms data idle timeout
         每轮启动/杀死（释放 NPU）
```

## 涉及的源文件

- `streaming_vad.py` — `record_on_voice_detection()` 函数（PTY 创建、os.read reader 线程、buf 正则提取、800ms 数据空闲超时、主循环）
- `run.sh` — 服务启动脚本，`LD_LIBRARY_PATH` 设置
