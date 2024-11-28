#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// ����ġ ���Ͽ��� ����ġ �б� �Լ�
void load_updated_weights(const char *weights_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *num_layers, int *layer_sizes)
{
    FILE *file = fopen(weights_file, "r");
    if (file == NULL)
    {
        perror("����ġ ������ �� �� �����ϴ�");
        exit(1);
    }

    char line[1024];
    int layer_index = -1;
    while (fgets(line, sizeof(line), file))
    {
        if (strncmp(line, "Layer", 5) == 0)
        {
            layer_index++;
            sscanf(line, "Layer %d Weights:", &layer_sizes[layer_index]);
            *num_layers = layer_index + 1;
        }
        else
        {
            char *token = strtok(line, " ");
            int weight_index = 0;
            while (token != NULL)
            {
                weights[layer_index][weight_index++] = atof(token);
                token = strtok(NULL, " ");
            }
        }
    }
    fclose(file);
}

// BMP �̹��� ���� �б� �Լ�
void read_bmp_image_1bit(const char *file_path, double *pixels, int input_size)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        perror("�̹��� ������ �� �� �����ϴ�");
        exit(1);
    }

    fseek(file, 54, SEEK_SET);
    unsigned char byte;
    int pixel_index = 0;
    while (fread(&byte, sizeof(unsigned char), 1, file) && pixel_index < input_size)
    {
        for (int i = 0; i < 8 && pixel_index < input_size; i++)
        {
            pixels[pixel_index++] = (byte >> (7 - i)) & 1;
        }
    }
    fclose(file);
}

// �����ͼ� �ε� �Լ�
void load_dataset(const char *csv_file, char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH], int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE], int *total_samples)
{
    FILE *file = fopen(csv_file, "r");
    if (file == NULL)
    {
        perror("CSV ������ �� �� �����ϴ�");
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file); // ��� �ǳʶٱ�

    int sample_index = 0;
    while (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");
        strcpy(dataset[sample_index], token);

        int label_index = 0;
        while ((token = strtok(NULL, ",")) != NULL)
        {
            labels[sample_index][label_index++] = atoi(token);
        }
        sample_index++;
    }
    *total_samples = sample_index;
    fclose(file);
}

// Sigmoid �Լ�
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// ReLU �Լ�
double relu(double x)
{
    return (x > 0) ? x : 0;
}

// Tanh �Լ�
double tanh_activation(double x)
{
    return tanh(x);
}

// Softmax �Լ� (���Ϳ� ���� ���)
void softmax(double *z, double *a, int size)
{
    double max_z = z[0];
    for (int i = 1; i < size; i++)
    {
        if (z[i] > max_z)
        {
            max_z = z[i];
        }
    }

    double sum_exp = 0;
    for (int i = 0; i < size; i++)
    {
        a[i] = exp(z[i] - max_z); // �ִ밪�� ���־� �����÷ο� ����
        sum_exp += a[i];
    }

    // ����ȭ
    for (int i = 0; i < size; i++)
    {
        a[i] /= sum_exp;
    }
}

// ��� ���� �Լ� (matrix * vector)
void mat_vec_mult(double *mat, double *vec, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += mat[i * cols + j] * vec[j];
        }
    }
}

// ������ �Լ�
void forward_propagation(double *input_data, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *layer_sizes, int num_layers, double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE], const char *activation_function, const char *output_activation_function)
{
    // ù ��° ���̾��� Ȱ��ȭ�� �Է� �����Ϳ� ����
    memcpy(activations[0], input_data, sizeof(double) * layer_sizes[0]);

    // �� ���̾ ���������� ���
    for (int i = 1; i < num_layers; i++)
    {
        double z[MAX_LAYER_SIZE] = {0}; // z ����
        double a[MAX_LAYER_SIZE] = {0}; // a ���� (Ȱ��ȭ�� ��)

        // ��� ���� (weights * activations)
        mat_vec_mult(weights[i - 1], activations[i - 1], z, layer_sizes[i], layer_sizes[i - 1]);

        // Ȱ��ȭ �Լ� ����
        if (i == num_layers - 1)
        { // ��� ���̾�
            if (strcmp(output_activation_function, "sigmoid") == 0)
            {
                // Sigmoid �Լ�
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = sigmoid(z[j]);
                }
            }
            else if (strcmp(output_activation_function, "softmax") == 0)
            {
                // Softmax �Լ�
                softmax(z, a, layer_sizes[i]);
            }
        }
        else
        { // ���� ���̾�
            if (strcmp(activation_function, "sigmoid") == 0)
            {
                // Sigmoid �Լ�
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = sigmoid(z[j]);
                }
            }
            else if (strcmp(activation_function, "relu") == 0)
            {
                // ReLU �Լ�
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = relu(z[j]);
                }
            }
            else if (strcmp(activation_function, "tanh") == 0)
            {
                // Tanh �Լ�
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = tanh_activation(z[j]);
                }
            }
        }

        // Ȱ��ȭ �� ����
        memcpy(activations[i], a, sizeof(double) * layer_sizes[i]);
    }
}

// �� �� �Լ�
void evaluate_model(const char *weights_file, const char *csv_file, int input_size, const char *output_log_file, const char *prediction_log_file)
{
    double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    int layer_sizes[MAX_LAYER_SIZE];
    int num_layers;

    // ����ġ ���Ͽ��� ����ġ �б�
    load_updated_weights(weights_file, weights, &num_layers, layer_sizes);

    // �����ͼ� �ε�
    char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH];
    int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE];
    int total_samples;
    load_dataset(csv_file, dataset, labels, &total_samples);

    FILE *label_log = fopen(output_log_file, "w");
    if (label_log == NULL)
    {
        perror("�� �� �α� ������ �� �� �����ϴ�");
        exit(1);
    }

    FILE *prediction_log = fopen(prediction_log_file, "w");
    if (prediction_log == NULL)
    {
        perror("���� ��� �α� ������ �� �� �����ϴ�");
        exit(1);
    }

    int correct_predictions = 0;
    double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE];

    for (int sample_index = 0; sample_index < total_samples; sample_index++)
    {
        double input_data[MAX_IMAGE_SIZE];
        read_bmp_image_1bit(dataset[sample_index], input_data, input_size);

        forward_propagation(input_data, weights, layer_sizes, num_layers, activations, "relu", "softmax");

        int predicted = 0;
        double max_value = activations[num_layers - 1][0];
        for (int i = 1; i < layer_sizes[num_layers - 1]; i++)
        {
            if (activations[num_layers - 1][i] > max_value)
            {
                max_value = activations[num_layers - 1][i];
                predicted = i;
            }
        }

        int target_class = 0;
        for (int i = 0; i < MAX_CHAR_LIST_SIZE; i++)
        {
            if (labels[sample_index][i] == 1)
            {
                target_class = i;
                break;
            }
        }

        // ������ �� ���� �� �� ���� ����
        fprintf(label_log, "�̹��� ���: %s\n", dataset[sample_index]);
        fprintf(label_log, "���� ��: %s, ��ǥ �ε���: %d\n", characters_list[target_class], target_class);
        fprintf(label_log, "���� ��: %s, ���� �ε���: %d\n", characters_list[predicted], predicted);
        // ���� �迭�� ���� �迭 ����
        fprintf(label_log, "���� �迭: ");
        for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
        {
            fprintf(label_log, "%d ", labels[sample_index][i]); // ���� �� (���� ���ڵ�)
        }
        fprintf(label_log, "\n");

        fprintf(label_log, "���� �迭: ");
        for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
        {
            fprintf(label_log, "%.4f ", activations[num_layers - 1][i]); // ���� Ȯ��
        }
        fprintf(label_log, "\n");
        fprintf(label_log, "====================\n");

        if (predicted == target_class)
        {
            fprintf(prediction_log, "����: %s, ����: %s - ������ �¾ҽ��ϴ�.\n", characters_list[target_class], characters_list[predicted]);
            correct_predictions++;
        }
        else
        {
            fprintf(prediction_log, "����: %s, ����: %s - ������ Ʋ�Ƚ��ϴ�.\n", characters_list[target_class], characters_list[predicted]);
        }
    }

    fclose(label_log);
    fclose(prediction_log);

    // �� ��Ȯ�� ���
    double accuracy = (double)correct_predictions / total_samples * 100;
    printf("\n�� ��Ȯ��: %.6f%%\n", accuracy);
}

int main()
{
    const char *weights_file = "C:/Users/asx12/OneDrive/���� ȭ��/�ΰ�����/all_training_results_plz/processed_weights_with_labels.txt";
    const char *csv_file = "C:/Users/asx12/OneDrive/���� ȭ��/�ΰ�����/dataset_labels_test.csv";
    const char *output_log_file = "C:/Users/asx12/OneDrive/���� ȭ��/�ΰ�����/label_comparison_output_all_c.txt";
    const char *prediction_log_file = "C:/Users/asx12/OneDrive/���� ȭ��/�ΰ�����/prediction_results_c.txt";
    int input_size = 64 * 64;

    evaluate_model(weights_file, csv_file, input_size, output_log_file, prediction_log_file);

    return 0;
}
