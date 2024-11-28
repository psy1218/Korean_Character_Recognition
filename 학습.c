#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 활성화 함수 (시그모이드)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 행렬 초기화 함수
void initialize_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // -1 ~ 1 사이의 난수
        }
    }
}

// 행렬 곱 함수
void matrix_multiply(double** A, int rows_A, int cols_A, double** B, int rows_B, int cols_B, double** result) {
    if (cols_A != rows_B) {
        printf("행렬 곱셈 오류: A의 열 수와 B의 행 수가 일치해야 합니다.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < cols_A; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 순전파 함수
void forward_propagation(double** layers[], double** weights[], int layer_num, int* node_counts, double** activations[]) {
    for (int i = 0; i < layer_num + 1; i++) {
        int current_layer_nodes = node_counts[i + 1];
        for (int j = 0; j < current_layer_nodes; j++) {
            double z = 0.0;
            for (int k = 0; k < node_counts[i]; k++) {
                z += weights[i][j][k] * layers[i][k][0];
            }
            activations[i + 1][j][0] = sigmoid(z);
        }
    }
}

// 오차 계산 함수 (MSE)
double calculate_error(double** output, double* target, int output_size) {
    double error = 0.0;
    for (int i = 0; i < output_size; i++) {
        error += pow(output[i][0] - target[i], 2);
    }
    return error / output_size;
}

// 메인 실행 함수
int main() {
    // 초기 설정
    int input_size = 64 * 64; // 입력층 크기 (64x64 이미지)
    int layer_num;
    printf("은닉층 개수를 입력하세요 (3~14): ");
    scanf("%d", &layer_num);

    if (layer_num < 3 || layer_num > 14) {
        printf("은닉층 개수는 3 이상, 14 이하이어야 합니다.\n");
        return 1;
    }

    // 레이어별 노드 개수 설정
    int node_counts[layer_num + 2];
    printf("\n각 레이어의 노드 개수를 입력하세요:\n");
    for (int i = 0; i < layer_num + 2; i++) {
        printf("Layer %d 노드 수: ", i + 1);
        scanf("%d", &node_counts[i]);
    }

    // 가중치 및 활성화 값 초기화
    double** weights[layer_num + 1];
    double** layers[layer_num + 2];
    double** activations[layer_num + 2];

    for (int i = 0; i < layer_num + 1; i++) {
        weights[i] = (double**)malloc(node_counts[i + 1] * sizeof(double*));
        for (int j = 0; j < node_counts[i + 1]; j++) {
            weights[i][j] = (double*)malloc(node_counts[i] * sizeof(double));
        }
        initialize_matrix(weights[i], node_counts[i + 1], node_counts[i]);
    }

    for (int i = 0; i < layer_num + 2; i++) {
        layers[i] = (double**)malloc(node_counts[i] * sizeof(double*));
        activations[i] = (double**)malloc(node_counts[i] * sizeof(double*));
        for (int j = 0; j < node_counts[i]; j++) {
            layers[i][j] = (double*)malloc(sizeof(double));
            activations[i][j] = (double*)malloc(sizeof(double));
        }
    }

    // 임의의 입력 데이터 및 목표값 생성
    for (int i = 0; i < node_counts[0]; i++) {
        layers[0][i][0] = ((double)rand() / RAND_MAX);
    }

    double target[node_counts[layer_num + 1]];
    for (int i = 0; i < node_counts[layer_num + 1]; i++) {
        target[i] = ((double)rand() / RAND_MAX);
    }

    // 순전파 및 결과 출력
    forward_propagation(layers, weights, layer_num, node_counts, activations);

    printf("\n--- 샘플 결과 ---\n");
    for (int i = 0; i < layer_num + 2; i++) {
        printf("Layer %d: ", i + 1);
        for (int j = 0; j < node_counts[i]; j++) {
            printf("%.4f ", activations[i][j][0]);
        }
        printf("\n");
    }

    double error = calculate_error(activations[layer_num + 1], target, node_counts[layer_num + 1]);
    printf("오차: %.4f\n", error);

    // 메모리 해제
    for (int i = 0; i < layer_num + 1; i++) {
        for (int j = 0; j < node_counts[i + 1]; j++) {
            free(weights[i][j]);
        }
        free(weights[i]);
    }

    for (int i = 0; i < layer_num + 2; i++) {
        for (int j = 0; j < node_counts[i]; j++) {
            free(layers[i][j]);
            free(activations[i][j]);
        }
        free(layers[i]);
        free(activations[i]);
    }

    return 0;
}
