/**
 * SenseVoice RKNN C++ 推理示例
 * 在 RK3588 上使用 RKNPU2 C API 进行语音情感识别
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <map>

#include "rknn_api.h"

// 情感标签映射
const std::map<int, std::string> EMOTION_LABELS = {
    {25001, "HAPPY"},
    {25002, "SAD"},
    {25003, "ANGRY"},
    {25004, "NEUTRAL"},
    {25005, "FEARFUL"},
    {25006, "SURPRISED"},
    {25007, "DISGUSTED"},
};

// 模型信息结构体
typedef struct {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;

// 读取模型文件
static int read_file(const char* path, char** data, size_t* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("Failed to open file: %s\n", path);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    *data = (char*)malloc(*size);
    if (!*data) {
        printf("Failed to allocate memory\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(*data, 1, *size, fp) != *size) {
        printf("Failed to read file\n");
        free(*data);
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}

// 初始化 RKNN 模型
int init_rknn_model(const char* model_path, rknn_app_context_t* app_ctx) {
    int ret;
    char* model_data;
    size_t model_size;
    
    // 读取模型文件
    ret = read_file(model_path, &model_data, &model_size);
    if (ret < 0) {
        printf("Failed to read model file\n");
        return -1;
    }
    
    // 初始化 RKNN 上下文
    ret = rknn_init(&app_ctx->ctx, (void*)model_data, model_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        free(model_data);
        return -1;
    }
    
    free(model_data);
    
    // 查询输入输出数量
    ret = rknn_query(app_ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret < 0) {
        printf("rknn_query failed! ret=%d\n", ret);
        return -1;
    }
    
    printf("Model input num: %d, output num: %d\n", app_ctx->io_num.n_input, app_ctx->io_num.n_output);
    
    // 查询输入属性
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        ret = rknn_query(app_ctx->ctx, RKNN_QUERY_INPUT_ATTR, &app_ctx->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query input attr failed! ret=%d\n", ret);
            return -1;
        }
        
        printf("Input %d: name=%s, shape=[%d, %d, %d, %d], fmt=%d, type=%d\n",
               i, app_ctx->input_attrs[i].name,
               app_ctx->input_attrs[i].dims[0], app_ctx->input_attrs[i].dims[1],
               app_ctx->input_attrs[i].dims[2], app_ctx->input_attrs[i].dims[3],
               app_ctx->input_attrs[i].fmt, app_ctx->input_attrs[i].type);
    }
    
    // 查询输出属性
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        ret = rknn_query(app_ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &app_ctx->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query output attr failed! ret=%d\n", ret);
            return -1;
        }
        
        printf("Output %d: name=%s, shape=[%d, %d, %d, %d], fmt=%d, type=%d\n",
               i, app_ctx->output_attrs[i].name,
               app_ctx->output_attrs[i].dims[0], app_ctx->output_attrs[i].dims[1],
               app_ctx->output_attrs[i].dims[2], app_ctx->output_attrs[i].dims[3],
               app_ctx->output_attrs[i].fmt, app_ctx->output_attrs[i].type);
    }
    
    return 0;
}

// 释放 RKNN 模型
int release_rknn_model(rknn_app_context_t* app_ctx) {
    if (app_ctx->input_attrs) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    
    if (app_ctx->ctx != 0) {
        rknn_destroy(app_ctx->ctx);
        app_ctx->ctx = 0;
    }
    
    return 0;
}

// 执行推理
int inference(rknn_app_context_t* app_ctx, float* input_data, int input_size, float** output_data, int* output_size) {
    int ret;
    
    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = input_size * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;
    
    ret = rknn_inputs_set(app_ctx->ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set failed! ret=%d\n", ret);
        return -1;
    }
    
    // 执行推理
    ret = rknn_run(app_ctx->ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed! ret=%d\n", ret);
        return -1;
    }
    
    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    
    ret = rknn_outputs_get(app_ctx->ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed! ret=%d\n", ret);
        return -1;
    }
    
    // 复制输出数据
    *output_size = outputs[0].size / sizeof(float);
    *output_data = (float*)malloc(outputs[0].size);
    memcpy(*output_data, outputs[0].buf, outputs[0].size);
    
    // 释放输出
    rknn_outputs_release(app_ctx->ctx, 1, outputs);
    
    return 0;
}

// 解码情感
std::string decode_emotion(float* output_data, int output_size) {
    // 找到最大值的索引
    int max_idx = 0;
    float max_val = output_data[0];
    
    for (int i = 1; i < output_size; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }
    
    // 查找情感标签
    auto it = EMOTION_LABELS.find(max_idx);
    if (it != EMOTION_LABELS.end()) {
        return it->second;
    }
    
    return "UNKNOWN";
}

// 获取当前时间 (毫秒)
double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <rknn_model> <input_data>\n", argv[0]);
        printf("  rknn_model: Path to RKNN model\n");
        printf("  input_data: Path to input data file (binary float32)\n");
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* input_path = argv[2];
    
    printf("SenseVoice RKNN Inference\n");
    printf("=========================\n");
    printf("Model: %s\n", model_path);
    printf("Input: %s\n", input_path);
    
    // 初始化模型
    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    
    int ret = init_rknn_model(model_path, &app_ctx);
    if (ret < 0) {
        printf("Failed to initialize RKNN model\n");
        return -1;
    }
    
    // 读取输入数据
    FILE* fp = fopen(input_path, "rb");
    if (!fp) {
        printf("Failed to open input file: %s\n", input_path);
        release_rknn_model(&app_ctx);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t input_size = ftell(fp) / sizeof(float);
    fseek(fp, 0, SEEK_SET);
    
    float* input_data = (float*)malloc(input_size * sizeof(float));
    fread(input_data, sizeof(float), input_size, fp);
    fclose(fp);
    
    printf("Input data size: %zu floats\n", input_size);
    
    // 执行推理
    float* output_data = NULL;
    int output_size = 0;
    
    double start_time = get_current_time();
    ret = inference(&app_ctx, input_data, input_size, &output_data, &output_size);
    double end_time = get_current_time();
    
    if (ret < 0) {
        printf("Inference failed\n");
        free(input_data);
        release_rknn_model(&app_ctx);
        return -1;
    }
    
    // 解码结果
    std::string emotion = decode_emotion(output_data, output_size);
    
    printf("\nResults:\n");
    printf("--------\n");
    printf("Emotion: %s\n", emotion.c_str());
    printf("Inference time: %.2f ms\n", end_time - start_time);
    
    // 清理
    free(input_data);
    free(output_data);
    release_rknn_model(&app_ctx);
    
    return 0;
}
