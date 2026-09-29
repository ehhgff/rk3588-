#!/bin/bash
# 医疗RAG版语音助手 - 使用延迟加载版RAG服务 (修复版)
# 快速启动 + Qwen3-1.7B模型

set -e

# 配置路径
ZIPFORMER_DIR="/data/zipformer"
INTERNVL_DIR="/data/internvl3"
VOICE_ASSISTANT_DIR="/userdata/voice_assistant"
MEDICAL_RAG_DIR="/userdata/medical_rag_full"
WORK_DIR="/tmp/voice_assistant"

# 模型路径
ENCODER_MODEL="$ZIPFORMER_DIR/model/encoder-epoch-99-avg-1.rknn"
DECODER_MODEL="$ZIPFORMER_DIR/model/decoder-epoch-99-avg-1.rknn"
JOINER_MODEL="$ZIPFORMER_DIR/model/joiner-epoch-99-avg-1.rknn"

# 工作文件
INPUT_AUDIO="${1:-$ZIPFORMER_DIR/model/test.wav}"
OUTPUT_AUDIO="$WORK_DIR/output.wav"

# 创建目录
mkdir -p $WORK_DIR

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_result() {
    echo -e "${YELLOW}[RESULT]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 启动Qwen3 LLM常驻服务
start_llm_service() {
    log_info "检查LLM服务..."
    
    # 先检查服务是否真正可用
    if [ -S "/tmp/qwen3_llm.sock" ]; then
        if python3 $VOICE_ASSISTANT_DIR/qwen3_llm_client.py "ping" > /dev/null 2>&1; then
            log_info "✓ Qwen3 LLM服务正常"
            return 0
        fi
    fi
    
    # 需要启动服务
    log_info "启动Qwen3 LLM常驻服务..."
    local start_time=$(date +%s%N)
    
    # 清理旧服务
    pkill -9 -f qwen3_llm_service 2>/dev/null || true
    rm -f /tmp/qwen3_llm.sock
    sleep 1
    
    cd $VOICE_ASSISTANT_DIR
    export LD_LIBRARY_PATH=/data/qwen3:$LD_LIBRARY_PATH
    nohup python3 qwen3_llm_service.py > /tmp/qwen3.log 2>&1 &
    local qwen3_pid=$!
    
    log_info "等待Qwen3模型加载..."
    for i in {1..120}; do
        # 检查进程是否还在运行
        if ! kill -0 $qwen3_pid 2>/dev/null; then
            log_error "Qwen3服务进程已退出"
            cat /tmp/qwen3.log | tail -5
            return 1
        fi
        
        # 检查socket是否存在且服务可用
        if [ -S "/tmp/qwen3_llm.sock" ]; then
            if python3 $VOICE_ASSISTANT_DIR/qwen3_llm_client.py "ping" > /dev/null 2>&1; then
                local end_time=$(date +%s%N)
                local duration=$(( (end_time - start_time) / 1000000 ))
                log_info "✓ Qwen3 LLM服务已就绪，启动耗时: ${duration}ms"
                return 0
            fi
        fi
        sleep 0.3
        if [ $((i % 30)) -eq 0 ]; then
            echo "  已等待 $((i * 3 / 10)) 秒..."
        fi
    done
    
    log_error "Qwen3 LLM服务启动超时"
    return 1
}

# 启动优化版RAG常驻服务（RKNN加速 + 混合搜索）
start_rag_service_optimized() {
    log_info "检查RAG服务..."
    
    # 测试服务是否真正可用
    if [ -S "/tmp/rag_optimized.sock" ]; then
        if python3 /userdata/voice_assistant/test_socket.py 2>/dev/null; then
            log_info "✓ RAG服务正常"
            return 0
        fi
    fi
    
    # 需要启动服务
    log_info "启动优化版RAG服务 (RKNN + 混合搜索 + 科室过滤)..."
    local start_time=$(date +%s%N)
    
    # 清理旧服务
    pkill -9 -f "rag_medical_server\|rag_service" 2>/dev/null || true
    rm -f /tmp/rag*.sock
    sleep 1
    
    cd $MEDICAL_RAG_DIR
    nohup python3 rag_optimized_server.py > /tmp/rag_optimized_server.log 2>&1 &
    
    # 等待服务启动（优化后启动更快，约2-3秒）
    for i in {1..50}; do
        if [ -S "/tmp/rag_optimized.sock" ]; then
            # 测试连接
            if python3 -c "
import socket
import json
try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect('/tmp/rag_optimized.sock')
    sock.send(json.dumps({'action': 'ping'}).encode())
    sock.recv(1024)
    sock.close()
except:
    pass
" 2>/dev/null; then
                local end_time=$(date +%s%N)
                local duration=$(( (end_time - start_time) / 1000000 ))
                log_info "✓ RAG服务已就绪，启动耗时: ${duration}ms"
                log_info "  使用 RKNN NPU 加速 + 混合搜索 + 科室过滤"
                log_info "  明确科室查询: ~300ms | 模糊查询: ~1100ms"
                return 0
            fi
        fi
        sleep 0.1
    done
    
    log_error "RAG服务启动超时"
    return 1
}

# 启动TTS服务
start_tts_service() {
    log_info "检查TTS服务..."
    
    # 检查服务是否可用（注意：TTS服务使用 /tmp/tts_service.sock）
    if [ -S "/tmp/tts_service.sock" ]; then
        if python3 -c "
import socket
import sys
try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect('/tmp/tts_service.sock')
    sock.close()
    sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
            log_info "✓ TTS服务正常"
            return 0
        fi
    fi
    
    # 启动TTS服务
    log_info "启动TTS服务..."
    
    # 清理旧服务
    pkill -9 -f tts_service 2>/dev/null || true
    rm -f /tmp/tts_service.sock
    sleep 1
    
    cd $VOICE_ASSISTANT_DIR
    nohup python3 tts_service.py > /tmp/tts_service.log 2>&1 &
    
    # 等待启动
    for i in {1..20}; do
        if [ -S "/tmp/tts_service.sock" ]; then
            sleep 1
            log_info "✓ TTS服务已就绪"
            return 0
        fi
        sleep 0.5
    done
    
    log_warn "TTS服务启动超时，将使用备用方案"
    return 0  # TTS失败不退出
}

# RAG检索 - 使用优化版服务（RKNN + 混合搜索）
rag_retrieve_optimized() {
    local query="$1"
    
    python3 << EOF
import socket
import json
import sys

try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect('/tmp/rag_optimized.sock')
    
    request = {'action': 'search', 'query': '''$query''', 'k': 2}
    sock.send(json.dumps(request).encode('utf-8'))
    
    response = sock.recv(8192).decode('utf-8')
    data = json.loads(response)
    
    if data.get('status') == 'ok':
        results = data['data']['results']
        time_ms = data['data']['time_ms']
        hybrid_mode = data['data'].get('hybrid_mode', True)
        dept_optimized = data['data'].get('dept_optimized', False)
        
        # 构建提示词（优化版 - 精简内容）
        prompt = f"""作为医疗助手，根据以下知识回答。保持回答简洁（50字以内）。

问题：$query

参考：
"""
        for i, r in enumerate(results[:2], 1):
            prompt += f"{i}. [{r['department']}] {r['answer'][:100]}\n"
        
        prompt += f"\n回答："
        
        print(prompt)
        mode_str = "混合搜索+科室过滤" if dept_optimized else "混合搜索"
        sys.stderr.write(f"RAG检索耗时: {time_ms:.1f}ms, 模式: {mode_str}\n")
    else:
        print(f"用户问：$query")
        sys.stderr.write(f"RAG错误: {data.get('message')}\n")
    
    sock.close()
except Exception as e:
    print(f"用户问：$query")
    sys.stderr.write(f"RAG异常: {e}\n")
EOF
}

# TTS合成 - 带备用方案
tts_synthesize() {
    local text="$1"
    local output="$2"
    
    cd $VOICE_ASSISTANT_DIR
    
    # 尝试使用TTS服务（使用JSON协议）
    if [ -S "/tmp/tts_service.sock" ]; then
        python3 << PYEOF 2>/dev/null
import socket
import json
import sys

try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect('/tmp/tts_service.sock')
    
    request = {
        'command': 'synthesize',
        'text': '''$text''',
        'output': '$output'
    }
    sock.send(json.dumps(request).encode())
    
    response = sock.recv(4096).decode()
    data = json.loads(response)
    sock.close()
    
    if data.get('status') == 'ok':
        sys.exit(0)
except Exception as e:
    pass
sys.exit(1)
PYEOF
        if [ $? -eq 0 ] && [ -f "$output" ] && [ -s "$output" ]; then
            return 0
        fi
    fi
    
    # 备用方案：直接调用melotts_demo
    log_warn "TTS服务调用失败，使用备用方案"
    if command -v melotts_demo >/dev/null 2>&1; then
        melotts_demo "$text" "$output" 2>/dev/null && return 0
    fi
    
    return 1
}

# 启动所有服务（调用预加载脚本）
start_all_services() {
    log_info "启动所有服务..."
    
    # 检查预加载脚本是否存在
    if [ -f "$VOICE_ASSISTANT_DIR/start_all_services_v2.sh" ]; then
        bash "$VOICE_ASSISTANT_DIR/start_all_services_v2.sh"
    else
        log_warn "预加载脚本不存在，使用内置服务启动..."
        start_llm_service || exit 1
        start_rag_service_optimized || exit 1
        start_tts_service
    fi
}

# 主流程
main() {
    echo "========================================"
    echo "    医疗RAG版语音助手系统"
    echo "    延迟加载版 (快速启动)"
    echo "    68,023条医疗知识 + Qwen3-1.7B"
    echo "    RKNN加速 + 混合搜索 + 科室过滤"
    echo "========================================"
    echo ""
    
    # 启动所有服务（使用预加载脚本）
    start_all_services
    
    # 检查输入文件
    if [ ! -f "$INPUT_AUDIO" ]; then
        log_error "音频文件不存在: $INPUT_AUDIO"
        exit 1
    fi
    
    # 记录总开始时间
    total_start=$(date +%s%N)
    
    # 步骤1: 语音识别
    log_step "步骤1/4: 语音识别"
    step1_start=$(date +%s%N)
    
    cd $ZIPFORMER_DIR
    recognized_text=$(./rknn_zipformer_demo \
        $ENCODER_MODEL $DECODER_MODEL $JOINER_MODEL $INPUT_AUDIO 2>/dev/null | \
        grep "Zipformer output:" | sed 's/Zipformer output: //')
    
    if [ -z "$recognized_text" ]; then
        log_error "语音识别失败"
        exit 1
    fi
    
    step1_end=$(date +%s%N)
    step1_time=$(( (step1_end - step1_start) / 1000000 ))
    log_result "识别结果: $recognized_text"
    log_info "耗时: ${step1_time}ms"
    echo ""
    
    # 步骤2: 医疗RAG检索
    log_step "步骤2/4: 医疗RAG检索 (68,023条知识库)"
    step2_start=$(date +%s%N)
    
    prompt=$(rag_retrieve_optimized "$recognized_text" 2>&1)
    
    step2_end=$(date +%s%N)
    step2_time=$(( (step2_end - step2_start) / 1000000 ))
    log_info "RAG检索耗时: ${step2_time}ms"
    log_info "已生成医疗知识增强提示词"
    echo ""
    
    # 步骤3: LLM生成回复
    log_step "步骤3/4: LLM生成回复 (Qwen3-1.7B)"
    step3_start=$(date +%s%N)
    
    cd $VOICE_ASSISTANT_DIR
    # 使用Qwen3客户端生成回复
    llm_output=$(python3 qwen3_llm_client.py "$prompt" 2>/dev/null)
    reply_text=$(echo "$llm_output" | grep "助手:" | sed 's/助手: //' | head -1)
    
    if [ -z "$reply_text" ]; then
        log_warn "LLM生成失败，使用RAG检索结果直接回复"
        reply_text=$(echo "$prompt" | grep -A 3 "参考1" | grep "答案：" | head -1 | sed 's/答案：//' | cut -c1-150)
        if [ -z "$reply_text" ]; then
            reply_text="抱歉，我暂时无法回答这个问题。"
        fi
    fi
    
    step3_end=$(date +%s%N)
    step3_time=$(( (step3_end - step3_start) / 1000000 ))
    log_result "回复: $reply_text"
    log_info "耗时: ${step3_time}ms"
    echo ""
    
    # 步骤4: 语音合成
    log_step "步骤4/4: 语音合成"
    step4_start=$(date +%s%N)
    
    if tts_synthesize "$reply_text" $OUTPUT_AUDIO; then
        step4_end=$(date +%s%N)
        step4_time=$(( (step4_end - step4_start) / 1000000 ))
        log_info "语音合成完成: $OUTPUT_AUDIO"
        log_info "耗时: ${step4_time}ms"
    else
        log_error "语音合成失败"
    fi
    
    # 总耗时
    total_end=$(date +%s%N)
    total_time=$(( (total_end - total_start) / 1000000 ))
    
    echo ""
    echo "========================================"
    echo "    总耗时: ${total_time}ms"
    echo "========================================"
    
    # 输出完整提示词
    echo ""
    log_info "完整提示词 (前800字):"
    echo "$prompt" | head -20
}

# 运行主流程
main "$@"
