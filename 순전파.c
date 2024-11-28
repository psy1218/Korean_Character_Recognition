#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 노드 개수 입력 및 첫 번째 레이어 노드 값 입력, 나머지 레이어 노드 배열 할당 
void input_nodes(int layer_num, int node_num[], double** nodes) {
    for (int i = 0; i < layer_num; i++) {
        printf("Layer %d의 node 개수를 입력하세요: ", i + 1);
        scanf("%d", &node_num[i]);  // 각 레이어의 노드 개수 입력받기

        if (i == 0) {
            // 첫 번째 레이어의 노드 값 입력받기
            nodes[i] = (double*)malloc(node_num[i] * sizeof(double));
            for (int j = 0; j < node_num[i]; j++) {
                printf("Layer %d의 노드 %d의 값을 입력하세요: ", i + 1, j + 1);
                scanf("%lf", &nodes[i][j]); 
            }
        }
        else {
            // 첫 번째 레이어 이후는 동적 배열로 할당  
            nodes[i] = (double*)malloc(node_num[i] * sizeof(double));
        }
    }
}


// 각 레이어 간의 가중치 입력받는 함수
void input_weights(int layer_num, int node_num[], double*** weights) {
    for (int i = 0; i < layer_num - 1; i++) { //가중치는 전체 레이어 수 - 1 
        int from_nodes = node_num[i];      // 현재 레이어의 노드 수
        int to_nodes = node_num[i + 1];    // 다음 레이어의 노드 수

        // from_nodes * to_nodes 크기의 가중치 2차원 배열 생성
        weights[i] = (double**)malloc(from_nodes * sizeof(double*));
        for (int j = 0; j < from_nodes; j++) {
            weights[i][j] = (double*)malloc(to_nodes * sizeof(double));
        }

        // 가중치 값 입력받기
        printf("\nLayer %d -> Layer %d 가중치 입력:\n", i + 1, i + 2);
        for (int j = 0; j < from_nodes; j++) {
            for (int k = 0; k < to_nodes; k++) {
                printf("가중치[%d][%d]: ", j + 1, k + 1);
                scanf("%lf", &weights[i][j][k]); 
            }
        }
    }
}

// 다음 레이어의 노드 값을 계산하는 함수
void next_nodes(int prev_nodes, int curr_nodes, double prev_values[], double** weights, double curr_values[]) {
    for (int j = 0; j < curr_nodes; j++) { //현재 구하고자 하는 레이어의 노드들
        curr_values[j] = 0;  // 동작배열 값 초기화
        for (int i = 0; i < prev_nodes; i++) { //이전 레이어의 노드들
            curr_values[j] += prev_values[i] * weights[i][j]; 

            //첫 번째 레이어를 제외한 나머지 레이어의 노드 값을 계산할 때는, 
            //현재 구하고자 하는 노드를 기준으로 이전 레이어의 모든 노드 값에 
            //해당 노드로 연결된 가중치를 곱하여 더해주면 된다.
        }
    }
}

// 시그모이드 활성 함수
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    int layer_num;

    printf("레이어 개수를 입력하세요: ");
    scanf("%d", &layer_num);

    // 각 레이어의 노드 개수를 저장할 배열
    int* node_num = (int*)malloc(layer_num * sizeof(int));

    // 각 레이어의 노드 값을 저장할 배열 
    double** nodes = (double**)malloc(layer_num * sizeof(double*));

    // 레이어의 노드 개수와 값 입력하기
    input_nodes(layer_num, node_num, nodes);

    // 각 레이어 간의 가중치를 저장할 배열
    double*** weights = (double***)malloc((layer_num - 1) * sizeof(double**));

    // 가중치 값 입력받기
    input_weights(layer_num, node_num, weights);

    // 각 레이어의 노드 값 계산 (첫 번째 레이어 제외)
    for (int i = 0; i < layer_num - 1; i++) {
        next_nodes(node_num[i], node_num[i + 1], nodes[i], weights[i], nodes[i + 1]);

        // 계산된 값에 시그모이드 함수 적용 (활성화 함수)
        for (int j = 0; j < node_num[i + 1]; j++) {
            nodes[i + 1][j] = sigmoid(nodes[i + 1][j]);
        }
    }

    // 결과 출력 (각 레이어의 노드 값 출력)
    for (int i = 0; i < layer_num; i++) {
        printf("\nLayer %d의 노드 값: ", i + 1);
        for (int j = 0; j < node_num[i]; j++) {
            printf("%lf ", nodes[i][j]); 
        }
        printf("\n");
    }


    for (int i = 0; i < layer_num; i++) {
        free(nodes[i]);  // 각 레이어의 노드 값 배열 해제
    }
    free(nodes);  // 레이어 배열 해제
    free(node_num);  // 노드 개수 배열 해제

    // 가중치 배열 메모리 해제
    for (int i = 0; i < layer_num - 1; i++) {
        for (int j = 0; j < node_num[i]; j++) {
            free(weights[i][j]);  // 가중치 배열 해제
        }
        free(weights[i]);
    }
    free(weights);

    return 0;
}
