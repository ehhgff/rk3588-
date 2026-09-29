#!/bin/bash
# 启动所有医疗RAG语音助手服务
# 包括：RAG服务、LLM服务、TTS服务

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查服务是否运行
check_service() {
    local socket=$1
    local name=$2
    
    if [ -S "$socket" ]; then
        return 0
    else
        return 1
    fi
}

# 启动LLM服务
start_llm_service() {
    log_step "启动 LLM 服务 (InternVL3-1B)..."
    
    if check_service "/tmp/voice_assistant.sock" "LLM"; then
        log_info "✓ LLM服务已在运行"
        return 0
    fi
    
    cd /userdata/voice_assistant
    export LD_LIBRARY_PATH=/data/internvl3
    
    setsid ./model_service_daemon > /tmp/llm_service.log 2>&1 &
    
    # 等待服务启动
    for i in {1..60}; do
        if check_service "/tmp/voice_assistant.sock" "LLM"; then
            sleep 1
            if ./llm_client ping > /dev/null 2>&1; then
                log_info "✓ LLM服务启动成功"
                return 0
            fi
        fi
        sleep 1
        if [ $((i % 10)) -eq 0 ]; then
            echo "  等待LLM服务... ${i}秒"
        fi
    done
    
    log_error "LLM服务启动超时"
    return 1
}

# 启动RAG服务
start_rag_service() {
    log_step "启动 RAG 服务 (68,023条医疗知识)..."
    
    if check_service "/tmp/rag_medical_full.sock" "RAG"; then
        log_info "✓ RAG服务已在运行"
        return 0
    fi
    
    cd /userdata/medical_rag_full
    
    setsid python3 rag_medical_server_full_v2.py start > /tmp/rag_full_server.log 2>&1 &
    
    # 等待服务启动
    for i in {1..90}; do
        if check_service "/tmp/rag_medical_full.sock" "RAG"; then
            log_info "✓ RAG服务启动成功"
            sleep 1
            return 0
        fi
        sleep 1
        if [ $((i % 15)) -eq 0 ]; then
            echo "  等待RAG服务... ${i}秒"
        fi
    done
    
    log_error "RAG服务启动超时"
    return 1
}

# 启动TTS服务
start_tts_service() {
    log_step "启动 TTS 服务 (MeloTTS)..."
    
    if check_service "/tmp/tts_service.sock" "TTS"; then
        log_info "✓ TTS服务已在运行"
        return 0
    fi
    
    cd /userdata/voice_assistant
    
    setsid python3 tts_service.py > /tmp/tts_service.log 2>&1 &
    
    # 等待服务启动
    for i in {1..30}; do
        if check_service "/tmp/tts_service.sock" "TTS"; then
            sleep 0.5
            if python3 tts_client.py ping > /dev/null 2>&1; then
                log_info "✓ TTS服务启动成功"
                return 0
            fi
        fi
        sleep 0.5
        if [ $((i % 10)) -eq 0 ]; then
            echo "  等待TTS服务... ${i}秒"
        fi
    done
    
    log_error "TTS服务启动超时"
    return 1
}

# 显示服务状态
show_status() {
    echo ""
    echo "========================================"
    echo "         服务状态检查"
    echo "========================================"
    
    # LLM服务
    if check_service "/tmp/voice_assistant.sock" "LLM"; then
        cd /userdata/voice_assistant
        if ./llm_client ping > /dev/null 2>&1; then
            echo -e "${GREEN}●${NC} LLM服务    (InternVL3-1B)    - 正常"
        else
            echo -e "${YELLOW}○${NC} LLM服务    (InternVL3-1B)    - 无响应"
        fi
    else
        echo -e "${RED}✗${NC} LLM服务    (InternVL3-1B)    - 未启动"
    fi
    
    # RAG服务
    if check_service "/tmp/rag_medical_full.sock" "RAG"; then
        echo -e "${GREEN}●${NC} RAG服务    (68,023条知识)    - 正常"
    else
        echo -e "${RED}✗${NC} RAG服务    (68,023条知识)    - 未启动"
    fi
    
    # TTS服务
    if check_service "/tmp/tts_service.sock" "TTS"; then
        cd /userdata/voice_assistant
        if python3 tts_client.py ping > /dev/null 2>&1; then
            echo -e "${GREEN}●${NC} TTS服务    (MeloTTS)         - 正常"
        else
            echo -e "${YELLOW}○${NC} TTS服务    (MeloTTS)         - 无响应"
        fi
    else
        echo -e "${RED}✗${NC} TTS服务    (MeloTTS)         - 未启动"
    fi
    
    echo "========================================"
}

# 停止所有服务
stop_all() {
    log_step "停止所有服务..."
    
    pkill -f model_service_daemon 2>/dev/null || true
    pkill -f "python3.*rag_medical_server" 2>/dev/null || true
    pkill -f "python3.*tts_service" 2>/dev/null || true
    
    rm -f /tmp/voice_assistant.sock /tmp/rag_medical_full.sock /tmp/tts_service.sock
    
    log_info "所有服务已停止"
}

# 主函数
main() {
    case "${1:-start}" in
        start)
            echo "========================================"
            echo "    启动医疗RAG语音助手服务"
            echo "========================================"
            echo ""
            
            # 启动所有服务
            start_llm_service || exit 1
            start_rag_service || exit 1
            start_tts_service || exit 1
            
            # 显示状态
            show_status
            
            echo ""
            log_info "所有服务启动完成！"
            echo ""
            echo "使用方式:"
            echo "  ./voice_assistant_full_rag_v2.sh <音频文件>"
            echo ""
            ;;
            
        stop)
            stop_all
            ;;
            
        status)
            show_status
            ;;
            
        restart)
            stop_all
            sleep 2
            main start
            ;;
            
        *)
            echo "用法: $0 {start|stop|status|restart}"
            exit 1
            ;;
    esac
}

main "$@"
