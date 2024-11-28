#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ��� ���� �Է� �� ù ��° ���̾� ��� �� �Է�, ������ ���̾� ��� �迭 �Ҵ� 
void input_nodes(int layer_num, int node_num[], double** nodes) {
    for (int i = 0; i < layer_num; i++) {
        printf("Layer %d�� node ������ �Է��ϼ���: ", i + 1);
        scanf("%d", &node_num[i]);  // �� ���̾��� ��� ���� �Է¹ޱ�

        if (i == 0) {
            // ù ��° ���̾��� ��� �� �Է¹ޱ�
            nodes[i] = (double*)malloc(node_num[i] * sizeof(double));
            for (int j = 0; j < node_num[i]; j++) {
                printf("Layer %d�� ��� %d�� ���� �Է��ϼ���: ", i + 1, j + 1);
                scanf("%lf", &nodes[i][j]); 
            }
        }
        else {
            // ù ��° ���̾� ���Ĵ� ���� �迭�� �Ҵ�  
            nodes[i] = (double*)malloc(node_num[i] * sizeof(double));
        }
    }
}


// �� ���̾� ���� ����ġ �Է¹޴� �Լ�
void input_weights(int layer_num, int node_num[], double*** weights) {
    for (int i = 0; i < layer_num - 1; i++) { //����ġ�� ��ü ���̾� �� - 1 
        int from_nodes = node_num[i];      // ���� ���̾��� ��� ��
        int to_nodes = node_num[i + 1];    // ���� ���̾��� ��� ��

        // from_nodes * to_nodes ũ���� ����ġ 2���� �迭 ����
        weights[i] = (double**)malloc(from_nodes * sizeof(double*));
        for (int j = 0; j < from_nodes; j++) {
            weights[i][j] = (double*)malloc(to_nodes * sizeof(double));
        }

        // ����ġ �� �Է¹ޱ�
        printf("\nLayer %d -> Layer %d ����ġ �Է�:\n", i + 1, i + 2);
        for (int j = 0; j < from_nodes; j++) {
            for (int k = 0; k < to_nodes; k++) {
                printf("����ġ[%d][%d]: ", j + 1, k + 1);
                scanf("%lf", &weights[i][j][k]); 
            }
        }
    }
}

// ���� ���̾��� ��� ���� ����ϴ� �Լ�
void next_nodes(int prev_nodes, int curr_nodes, double prev_values[], double** weights, double curr_values[]) {
    for (int j = 0; j < curr_nodes; j++) { //���� ���ϰ��� �ϴ� ���̾��� ����
        curr_values[j] = 0;  // ���۹迭 �� �ʱ�ȭ
        for (int i = 0; i < prev_nodes; i++) { //���� ���̾��� ����
            curr_values[j] += prev_values[i] * weights[i][j]; 

            //ù ��° ���̾ ������ ������ ���̾��� ��� ���� ����� ����, 
            //���� ���ϰ��� �ϴ� ��带 �������� ���� ���̾��� ��� ��� ���� 
            //�ش� ���� ����� ����ġ�� ���Ͽ� �����ָ� �ȴ�.
        }
    }
}

// �ñ׸��̵� Ȱ�� �Լ�
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    int layer_num;

    printf("���̾� ������ �Է��ϼ���: ");
    scanf("%d", &layer_num);

    // �� ���̾��� ��� ������ ������ �迭
    int* node_num = (int*)malloc(layer_num * sizeof(int));

    // �� ���̾��� ��� ���� ������ �迭 
    double** nodes = (double**)malloc(layer_num * sizeof(double*));

    // ���̾��� ��� ������ �� �Է��ϱ�
    input_nodes(layer_num, node_num, nodes);

    // �� ���̾� ���� ����ġ�� ������ �迭
    double*** weights = (double***)malloc((layer_num - 1) * sizeof(double**));

    // ����ġ �� �Է¹ޱ�
    input_weights(layer_num, node_num, weights);

    // �� ���̾��� ��� �� ��� (ù ��° ���̾� ����)
    for (int i = 0; i < layer_num - 1; i++) {
        next_nodes(node_num[i], node_num[i + 1], nodes[i], weights[i], nodes[i + 1]);

        // ���� ���� �ñ׸��̵� �Լ� ���� (Ȱ��ȭ �Լ�)
        for (int j = 0; j < node_num[i + 1]; j++) {
            nodes[i + 1][j] = sigmoid(nodes[i + 1][j]);
        }
    }

    // ��� ��� (�� ���̾��� ��� �� ���)
    for (int i = 0; i < layer_num; i++) {
        printf("\nLayer %d�� ��� ��: ", i + 1);
        for (int j = 0; j < node_num[i]; j++) {
            printf("%lf ", nodes[i][j]); 
        }
        printf("\n");
    }


    for (int i = 0; i < layer_num; i++) {
        free(nodes[i]);  // �� ���̾��� ��� �� �迭 ����
    }
    free(nodes);  // ���̾� �迭 ����
    free(node_num);  // ��� ���� �迭 ����

    // ����ġ �迭 �޸� ����
    for (int i = 0; i < layer_num - 1; i++) {
        for (int j = 0; j < node_num[i]; j++) {
            free(weights[i][j]);  // ����ġ �迭 ����
        }
        free(weights[i]);
    }
    free(weights);

    return 0;
}
