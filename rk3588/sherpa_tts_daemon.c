/*
 * sherpa_tts_daemon.c - TTS 常驻服务
 *
 * 使用方法:
 *   1. 启动服务:
 *      ./sherpa_tts_daemon /path/to/model_dir /tmp/tts_pipe
 *
 *   2. 发送文本:
 *      echo "你好世界" > /tmp/tts_pipe
 *
 *   3. 输出: /tmp/tts_out_%d.wav（推理只花~0.5s，不含模型加载）
 *
 * 编译 (交叉编译):
 *   aarch64-linux-gnu-gcc sherpa_tts_daemon.c \
 *       -o sherpa_tts_daemon \
 *       -I/userdata/sherpa-onnx/install/include \
 *       -L/userdata/sherpa-onnx/install/lib \
 *       -lsherpa-onnx-c-api -lcargs -lonnxruntime \
 *       -Wl,-rpath,/userdata/sherpa-onnx/install/lib \
 *       -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>
#include <time.h>

#include "sherpa-onnx/c-api/c-api.h"

static volatile int keep_running = 1;

void handle_signal(int sig) {
    (void)sig;
    keep_running = 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <pipe_path>\n", argv[0]);
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  # 终端1: mkfifo /tmp/tts_pipe\n");
        fprintf(stderr, "  # 终端1: %s /userdata/sherpa-onnx/matcha-zh-baker /tmp/tts_pipe\n", argv[0]);
        fprintf(stderr, "  # 终端2: echo '你好世界' > /tmp/tts_pipe\n");
        return 1;
    }

    const char *model_dir = argv[1];
    const char *pipe_path = argv[2];
    int output_idx = 0;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    // 构建模型文件路径
    char acoustic_path[512], vocoder_path[512];
    char lexicon_path[512], tokens_path[512], out_path[512];

    snprintf(acoustic_path, sizeof(acoustic_path), "%s/model-steps-3.onnx", model_dir);
    snprintf(vocoder_path, sizeof(vocoder_path), "%s/vocos-22khz-univ.onnx", model_dir);
    snprintf(lexicon_path, sizeof(lexicon_path), "%s/lexicon.txt", model_dir);
    snprintf(tokens_path, sizeof(tokens_path), "%s/tokens.txt", model_dir);

    // 检查文件存在
    if (access(acoustic_path, F_OK) != 0) {
        fprintf(stderr, "Error: %s not found\n", acoustic_path);
        return 1;
    }
    if (access(vocoder_path, F_OK) != 0) {
        fprintf(stderr, "Error: %s not found\n", vocoder_path);
        return 1;
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "========================================\n");
    fprintf(stdout, " sherpa-onnx TTS Daemon\n");
    fprintf(stdout, "========================================\n");
    fprintf(stdout, " Model dir : %s\n", model_dir);
    fprintf(stdout, " Pipe path : %s\n", pipe_path);
    fprintf(stdout, "\n");

    // ---- 1. 创建 TTS 引擎（只加载一次） ----
    fprintf(stdout, " [1/2] 加载模型...\n");

    SherpaOnnxOfflineTtsConfig tts_config;
    memset(&tts_config, 0, sizeof(tts_config));

    SherpaOnnxOfflineTtsModelConfig *model_config = &tts_config.model;

    // Matcha 模型配置
    model_config->matcha.acoustic_model = acoustic_path;
    model_config->matcha.vocoder = vocoder_path;
    model_config->matcha.lexicon = lexicon_path;
    model_config->matcha.tokens = tokens_path;
    model_config->matcha.noise_scale = 0.333f;
    model_config->matcha.length_scale = 1.0f;

    model_config->num_threads = 4;
    model_config->provider = "cpu";
    model_config->debug = 0;

    tts_config.max_num_sentences = 2;
    tts_config.silence_scale = 0.2f;

    double t0 = now_ms();
    const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&tts_config);
    double t1 = now_ms();

    if (!tts) {
        fprintf(stderr, "Error: Failed to create TTS engine\n");
        return 1;
    }

    int sample_rate = SherpaOnnxOfflineTtsSampleRate(tts);
    fprintf(stdout, " [OK] 模型加载完成: %.1f ms\n", t1 - t0);
    fprintf(stdout, "      Sample rate: %d Hz\n", sample_rate);
    fprintf(stdout, "      Speakers: %d\n", SherpaOnnxOfflineTtsNumSpeakers(tts));
    fprintf(stdout, "\n");

    // ---- 2. 创建管道 ----
    unlink(pipe_path);
    if (mkfifo(pipe_path, 0666) != 0) {
        perror("mkfifo");
        SherpaOnnxDestroyOfflineTts(tts);
        return 1;
    }

    fprintf(stdout, " [2/2] 等待输入...\n");
    fprintf(stdout, "      echo 'text' > %s\n", pipe_path);
    fprintf(stdout, "      输出: /tmp/tts_out_<idx>.wav\n");
    fprintf(stdout, "      Ctrl+C 退出\n");
    fprintf(stdout, "========================================\n\n");

    // ---- 3. 主循环：从管道读取文本，生成语音 ----
    while (keep_running) {
        int fd = open(pipe_path, O_RDONLY);
        if (fd < 0) {
            if (keep_running) perror("open pipe");
            break;
        }

        char text[4096];
        ssize_t n = read(fd, text, sizeof(text) - 1);
        close(fd);

        if (n <= 0) continue;
        text[n] = '\0';

        // 去除末尾换行
        char *nl = strchr(text, '\n');
        if (nl) *nl = '\0';
        if (strlen(text) == 0) continue;

        fprintf(stdout, " [TTS] \"%s\"\n", text);

        // 生成音频
        SherpaOnnxGenerationConfig gen_cfg;
        memset(&gen_cfg, 0, sizeof(gen_cfg));
        gen_cfg.sid = 0;
        gen_cfg.speed = 1.0f;
        gen_cfg.silence_scale = 0.2f;

        double infer_t0 = now_ms();
        const SherpaOnnxGeneratedAudio *audio =
            SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &gen_cfg, NULL, NULL);
        double infer_t1 = now_ms();

        if (!audio || audio->n == 0) {
            fprintf(stderr, " [ERR] 生成失败\n");
            continue;
        }

        double infer_ms = infer_t1 - infer_t0;
        double audio_dur = (double)audio->n / audio->sample_rate;
        double rtf = infer_ms / 1000.0 / audio_dur;

        fprintf(stdout, "      推理: %.1f ms | 音频: %.2f s | RTF: %.3f\n",
                infer_ms, audio_dur, rtf);

        // 保存 WAV
        snprintf(out_path, sizeof(out_path), "/tmp/tts_out_%d.wav", output_idx++);
        if (SherpaOnnxWriteWave(audio->samples, audio->n,
                                audio->sample_rate, out_path)) {
            fprintf(stdout, "      输出: %s\n", out_path);
        }

        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        fprintf(stdout, "      等待下一条...\n\n");
    }

    // ---- 清理 ----
    fprintf(stdout, "\n清理中...\n");
    SherpaOnnxDestroyOfflineTts(tts);
    unlink(pipe_path);
    fprintf(stdout, "已退出\n");

    return 0;
}