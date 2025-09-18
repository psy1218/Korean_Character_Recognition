#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>  // isspace() �Լ��� ����ϱ� ���� ��� ����

#define MAX_LAYER_SIZE 256
#define MAX_CHAR_LIST_SIZE 115
#define MAX_IMAGE_SIZE 4096
#define MAX_PATH_LENGTH 256

// ���� ���ڵ��� ������ ���� ���� ����Ʈ
const char characters_list[MAX_CHAR_LIST_SIZE][4] = {
    "��", "û", "��", "��", "ȫ", "��", "��", "��", "��", "��", "��", "��", "ȣ", "��", "��", "��", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "��", "��", "��", "��", "ȭ", "õ", "��", "��", "��", "��", "��", "��", "ī", "��",
    "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "Į", "��", "��", "��", "��", "��", "��", "��", "��", "��", "��"};

void load_updated_weights(const char *weights_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *num_layers, int *layer_sizes, const char *debug_log_file) {
    FILE *file = fopen(weights_file, "r");
    if (file == NULL) {
        perror("����ġ ������ �� �� �����ϴ�");
        exit(1);
    }

    FILE *debug_log = fopen(debug_log_file, "w");
    if (debug_log == NULL) {
        perror("����� �α� ������ �� �� �����ϴ�");
        fclose(file);
        exit(1);
    }

    char line[4096];
    int layer_index = -1;
    int weight_index = 0;

    while (fgets(line, sizeof(line), file)) {
        // ���� ���� ����
        line[strcspn(line, "\n")] = '\0';

        // ���̾� ������ �д� �κ�
        if (strncmp(line, "Layer", 5) == 0) {
            layer_index++;
            weight_index = 0;  // ���ο� ���̾ ���۵Ǹ� �ε����� 0���� �ʱ�ȭ

            // "Layer 1 Weights:" ������ ���ڿ����� ���̾� ��ȣ�� �Ľ��մϴ�.
            int temp_layer_index;
            if (sscanf(line, "Layer %d Weights:", &temp_layer_index) == 1) {
                // ���̾� ũ�⸦ �Ľ��ϰų� �⺻�� ����
                if (sscanf(line, "Layer %*d Weights: %d", &layer_sizes[layer_index]) != 1) {
                    // ���� ���̾� ũ�Ⱑ ���Ͽ� ��õ��� ���� ���, �⺻�� ���
                    layer_sizes[layer_index] = MAX_LAYER_SIZE;  // �⺻�� ���� (�ʿ� �� ����)
                }
            } else {
                fprintf(stderr, "���̾� ������ �д� �� �����߽��ϴ�: %s\n", line);
                fclose(file);
                fclose(debug_log);
                exit(1);
            }

            *num_layers = layer_index + 1;
        } else {
            // ���� ���ڿ��� �Ľ��ϱ� ���� ������
            char *token = line;
            while (*token) {
                // ������ �ǳʶ�
                while (*token && isspace(*token)) {
                    token++;
                }

                // ���� �б�
                double value;
                if (sscanf(token, "%lf", &value) == 1) {
                    // ���� ���̾��� ũ�⸦ �ʰ��ϴ��� Ȯ��
                    if (weight_index >= layer_sizes[layer_index]) {
                        fprintf(stderr, "����ġ �ε��� �ʰ�: Layer %d, Weight %d\n", layer_index, weight_index);
                        fclose(file);
                        fclose(debug_log);
                        exit(1);
                    }

                    weights[layer_index][weight_index++] = value;
                    fprintf(debug_log, "Layer %d, Weight %d: %f\n", layer_index, weight_index - 1, value);

                    // ���� ���ڷ� �̵�
                    while (*token && !isspace(*token)) {
                        token++;
                    }
                } else {
                    break;  // ���ڰ� �ƴ� ��� ���� Ż��
                }
            }
        }
    }

    fclose(file);
    fclose(debug_log);
}

// ����ġ ���� �Լ� (���Ϸ� ����)
void save_weights(const char *output_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int num_layers, int *layer_sizes) {
    FILE *file = fopen(output_file, "w");
    if (file == NULL) {
        perror("����ġ ���� ������ �� �� �����ϴ�");
        exit(1);
    }

    for (int i = 0; i < num_layers; i++) {
        fprintf(file, "Layer %d Weights:\n", i);
        for (int j = 0; j < layer_sizes[i]; j++) {
            fprintf(file, "%.6f ", weights[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// �����ͼ� �ε� �Լ�
void load_dataset(const char *csv_file, char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH], int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE], int *total_samples) {
    FILE *file = fopen(csv_file, "r");
    if (file == NULL) {
        perror("CSV ������ �� �� �����ϴ�");
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file); // ��� �ǳʶٱ�

    int sample_index = 0;
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        strcpy(dataset[sample_index], token);

        int label_index = 0;
        while ((token = strtok(NULL, ",")) != NULL) {
            labels[sample_index][label_index++] = atoi(token);
        }
        sample_index++;
    }
    *total_samples = sample_index;
    fclose(file);
}

// BMP �̹��� ���� �б� �Լ�
void read_bmp_image_1bit(const char *file_path, double *pixels, int input_size) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("�̹��� ������ �� �� �����ϴ�");
        exit(1);
    }

    fseek(file, 54, SEEK_SET);
    unsigned char byte;
    int pixel_index = 0;
    while (fread(&byte, sizeof(unsigned char), 1, file) && pixel_index < input_size) {
        for (int i = 0; i < 8 && pixel_index < input_size; i++) {
            pixels[pixel_index++] = (byte >> (7 - i)) & 1;
        }
    }
    fclose(file);
}

// ������ �Լ�
void forward_propagation(double *input_data, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *layer_sizes, int num_layers, double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE], const char *activation_function, const char *output_activation_function) {
    // ù ��° ���̾��� Ȱ��ȭ�� �Է� �����Ϳ� ����
    memcpy(activations[0], input_data, sizeof(double) * layer_sizes[0]);

    // �� ���̾ ���������� ���
    for (int i = 1; i < num_layers; i++) {
        double z[MAX_LAYER_SIZE] = {0}; // z ����
        double a[MAX_LAYER_SIZE] = {0}; // a ���� (Ȱ��ȭ�� ��)

        // ��� ���� (weights * activations)
        for (int j = 0; j < layer_sizes[i]; j++) {
            z[j] = 0;
            for (int k = 0; k < layer_sizes[i - 1]; k++) {
                z[j] += weights[i - 1][j * layer_sizes[i - 1] + k] * activations[i - 1][k];
            }
        }

        // Ȱ��ȭ �Լ� ����
        if (i == num_layers - 1) { // ��� ���̾�
            if (strcmp(output_activation_function, "sigmoid") == 0) {
                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] = 1.0 / (1.0 + exp(-z[j]));
                }
            } else if (strcmp(output_activation_function, "softmax") == 0) {
                double max_z = z[0];
                for (int j = 1; j < layer_sizes[i]; j++) {
                    if (z[j] > max_z) {
                        max_z = z[j];
                    }
                }

                double sum_exp = 0;
                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] = exp(z[j] - max_z);
                    sum_exp += a[j];
                }

                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] /= sum_exp;
                }
            }
        } else { // ���� ���̾�
            if (strcmp(activation_function, "sigmoid") == 0) {
                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] = 1.0 / (1.0 + exp(-z[j]));
                }
            } else if (strcmp(activation_function, "relu") == 0) {
                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] = (z[j] > 0) ? z[j] : 0;
                }
            } else if (strcmp(activation_function, "tanh") == 0) {
                for (int j = 0; j < layer_sizes[i]; j++) {
                    a[j] = tanh(z[j]);
                }
            }
        }

        // Ȱ��ȭ �� ����
        memcpy(activations[i], a, sizeof(double) * layer_sizes[i]);
    }
}

// �� �� �Լ�
void evaluate_model(const char *weights_file, const char *csv_file, int input_size, const char *output_log_file, const char *prediction_log_file, const char *forward_log_file, const char *weights_output_file, const char *debug_log_file) {
    double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    int layer_sizes[MAX_LAYER_SIZE];
    int num_layers;

    // ����ġ ���Ͽ��� ����ġ �б�
    load_updated_weights(weights_file, weights, &num_layers, layer_sizes, debug_log_file);

    // ����ġ �����ϱ� (����ġ Ȯ�ο�)
    save_weights(weights_output_file, weights, num_layers, layer_sizes);

    // �����ͼ� �ε�
    char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH];
    int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE];
    int total_samples;
    load_dataset(csv_file, dataset, labels, &total_samples);

    FILE *label_log = fopen(output_log_file, "w");
    if (label_log == NULL) {
        perror("�� �� �α� ������ �� �� �����ϴ�");
        exit(1);
    }

    FILE *prediction_log = fopen(prediction_log_file, "w");
    if (prediction_log == NULL) {
        perror("���� ��� �α� ������ �� �� �����ϴ�");
        exit(1);
    }

    // ������ �α� ���� ����
    FILE *forward_log = fopen(forward_log_file, "w");
    if (forward_log == NULL) {
        perror("������ �α� ������ �� �� �����ϴ�");
        exit(1);
    }

    int correct_predictions = 0;
    double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE];

    for (int sample_index = 0; sample_index < total_samples; sample_index++) {
        double input_data[MAX_IMAGE_SIZE];
        read_bmp_image_1bit(dataset[sample_index], input_data, input_size);

        // ������ ���
        fprintf(forward_log, "=== Sample %d ===\n", sample_index + 1);
        fprintf(forward_log, "�̹��� ���: %s\n", dataset[sample_index]);
        forward_propagation(input_data, weights, layer_sizes, num_layers, activations, "relu", "softmax");

        int predicted = 0;
        double max_value = activations[num_layers - 1][0];
        for (int i = 1; i < layer_sizes[num_layers - 1]; i++) {
            if (activations[num_layers - 1][i] > max_value) {
                max_value = activations[num_layers - 1][i];
                predicted = i;
            }
        }

        int target_class = 0;
        for (int i = 0; i < MAX_CHAR_LIST_SIZE; i++) {
            if (labels[sample_index][i] == 1) {
                target_class = i;
                break;
            }
        }

        if (predicted == target_class) {
            correct_predictions++;
        }
    }

    fclose(label_log);
    fclose(prediction_log);
    fclose(forward_log);

    // �� ��Ȯ�� ���
    double accuracy = (double)correct_predictions / total_samples * 100;
    printf("\n�� ��Ȯ��: %.6f%%\n", accuracy);
}

int main() {
    const char
