#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_PATH_LENGTH 256
#define MAX_NODES 100

// 시그모이드 활성화 함수
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// ReLU 활성화 함수
double relu(double x) {
    return x > 0 ? x : 0;
}

// Softmax 활성화 함수
void softmax(double* input, int length) {
    double max_value = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_value) {
            max_value = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max_value);
        sum += input[i];
    }

    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

// 가중치 파일에서 가중치 읽기 함수
void load_weights(const char* weights_file, int layer_num, int node_num[], double*** weights) {
    FILE* file = fopen(weights_file, "r");
    if (file == NULL) {
        printf("가중치 파일을 열 수 없습니다.\n");
        exit(1);
    }

    for (int i = 0; i < layer_num - 1; i++) {
        int from_nodes = node_num[i];
        int to_nodes = node_num[i + 1];

        weights[i] = (double**)malloc(from_nodes * sizeof(double*));
        for (int j = 0; j < from_nodes; j++) {
            weights[i][j] = (double*)malloc(to_nodes * sizeof(double));
            for (int k = 0; k < to_nodes; k++) {
                fscanf(file, "%lf", &weights[i][j][k]);
            }
        }
    }

    fclose(file);
}

// 가중치 파일에서 읽은 내용을 별도의 파일로 저장하는 함수
void save_weights(const char* output_file, int layer_num, int node_num[], double*** weights) {
    FILE* file = fopen(output_file, "w");
    if (file == NULL) {
        printf("가중치 저장 파일을 열 수 없습니다.\n");
        exit(1);
    }

    for (int i = 0; i < layer_num - 1; i++) {
        fprintf(file, "Layer %d Weights:\n", i + 1);
        int from_nodes = node_num[i];
        int to_nodes = node_num[i + 1];
        for (int j = 0; j < from_nodes; j++) {
            for (int k = 0; k < to_nodes; k++) {
                fprintf(file, "%lf ", weights[i][j][k]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// BMP 이미지 파일 읽기 함수
void read_bmp_image(const char* file_path, int input_size, double* input_data) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        printf("이미지 파일을 열 수 없습니다.\n");
        exit(1);
    }

    fseek(file, 54, SEEK_SET);  // BMP 파일 헤더를 건너뜀
    unsigned char pixel_data;
    for (int i = 0; i < input_size / 8; i++) {
        fread(&pixel_data, sizeof(unsigned char), 1, file);
        for (int bit = 0; bit < 8; bit++) {
            input_data[i * 8 + bit] = (pixel_data & (1 << (7 - bit))) ? 1.0 : 0.0;
        }
    }

    fclose(file);
}

// 순전파 계산 함수
void forward_propagation(int layer_num, int node_num[], double** nodes, double*** weights) {
    for (int i = 0; i < layer_num - 1; i++) {
        int prev_nodes = node_num[i];
        int curr_nodes = node_num[i + 1];

        for (int j = 0; j < curr_nodes; j++) {
            nodes[i + 1][j] = 0;
            for (int k = 0; k < prev_nodes; k++) {
                nodes[i + 1][j] += nodes[i][k] * weights[i][k][j];
            }
            // 은닉층에서는 ReLU 활성화 함수 적용
            if (i < layer_num - 2) {
                nodes[i + 1][j] = relu(nodes[i + 1][j]);
            }
        }
        // 출력층에서는 Softmax 활성화 함수 적용
        if (i == layer_num - 2) {
            softmax(nodes[i + 1], curr_nodes);
        }
    }
}

// 데이터셋 로드 함수
void load_dataset(const char* csv_file, char image_paths[][MAX_PATH_LENGTH], int labels[][115], int* total_samples) {
    FILE* file = fopen(csv_file, "r");
    if (file == NULL) {
        printf("데이터셋 파일을 열 수 없습니다.\n");
        exit(1);
    }

    char line[1024];
    *total_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        strcpy(image_paths[*total_samples], token);
        int index = 0;
        while ((token = strtok(NULL, ",")) != NULL) {
            labels[*total_samples][index++] = atoi(token);
        }
        (*total_samples)++;
    }

    fclose(file);
}

// 예측 및 평가 함수
void evaluate_model(const char* weights_file, const char* csv_file, int layer_num, int node_num[], double** nodes, double*** weights) {
    // 가중치 읽기
    load_weights(weights_file, layer_num, node_num, weights);

    // 데이터셋 로드
    char image_paths[MAX_NODES][MAX_PATH_LENGTH];
    int labels[MAX_NODES][115];
    int total_samples;
    load_dataset(csv_file, image_paths, labels, &total_samples);

    int correct_predictions = 0;

    // 각 샘플에 대해 평가
    for (int i = 0; i < total_samples; i++) {
        // 입력 데이터 읽기
        read_bmp_image(image_paths[i], node_num[0], nodes[0]);

        // 순전파 수행
        forward_propagation(layer_num, node_num, nodes, weights);

        // 예측값 계산
        int predicted_class = 0;
        double max_value = nodes[layer_num - 1][0];
        for (int j = 1; j < node_num[layer_num - 1]; j++) {
            if (nodes[layer_num - 1][j] > max_value) {
                max_value = nodes[layer_num - 1][j];
                predicted_class = j;
            }
        }

        // 실제 라벨과 비교
        int target_class = 0;
        for (int j = 0; j < 115; j++) {
            if (labels[i][j] == 1) {
                target_class = j;
                break;
            }
        }

        if (predicted_class == target_class) {
            correct_predictions++;
        }
    }

    // 평가 정확도 출력
    double accuracy = (double)correct_predictions / total_samples * 100;
    printf("\n평가 정확도: %.6f%%\n", accuracy);
}

int main() {
    int layer_num;

    printf("레이어 개수를 입력하세요 (최대 %d): ", MAX_NODES);
    scanf("%d", &layer_num);
    if (layer_num > MAX_NODES) {
        printf("레이어 개수는 최대 %d까지 가능합니다. 프로그램을 종료합니다.\n", MAX_NODES);
        exit(1);
    }

    int* node_num = (int*)malloc(layer_num * sizeof(int));
    double** nodes = (double**)malloc(layer_num * sizeof(double*));
    double*** weights = (double***)malloc((layer_num - 1) * sizeof(double**));

    for (int i = 0; i < layer_num; i++) {
        nodes[i] = (double*)malloc(MAX_NODES * sizeof(double));
    }

    // 가중치 파일 경로 및 데이터셋 파일 경로 설정
    const char* weights_file = "weights.txt";
    const char* csv_file = "dataset.csv";

    // 모델 평가
    evaluate_model(weights_file, csv_file, layer_num, node_num, nodes, weights);

    // 메모리 해제
    for (int i = 0; i < layer_num; i++) {
        free(nodes[i]);
    }
    free(nodes);
    free(node_num);

    for (int i = 0; i < layer_num - 1; i++) {
        for (int j = 0; j < MAX_NODES; j++) {
            free(weights[i][j]);
        }
        free(weights[i]);
    }
    free(weights);

    return 0;
}
