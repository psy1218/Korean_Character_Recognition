#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LAYER_SIZE 256
#define MAX_CHAR_LIST_SIZE 115
#define MAX_IMAGE_SIZE 4096
#define MAX_PATH_LENGTH 256

// 원핫 인코딩한 순서에 따른 글자 리스트
const char characters_list[MAX_CHAR_LIST_SIZE][4] = {
    "충", "청", "남", "도", "홍", "성", "군", "읍", "기", "길", "숙", "사", "호", "동", "박", "소", "윤", "김", "은", "옥",
    "영", "수", "종", "헌", "조", "인", "의", "경", "봉", "화", "천", "광", "역", "시", "연", "구", "아", "카", "데",
    "미", "로", "춘", "이", "는", "듣", "만", "하", "여", "가", "슴", "설", "레", "말", "다", "너", "두", "손", "을", "에",
    "대", "고", "물", "방", "같", "심", "장", "들", "어", "보", "라", "피", "끓", "뛰", "노", "거", "선", "관", "힘", "있",
    "것", "류", "를", "꾸", "며", "내", "려", "온", "력", "바", "투", "명", "되", "얼", "음", "과", "으", "지", "혜", "날",
    "우", "나", "갑", "속", "든", "칼", "니", "더", "면", "간", "마", "쓸", "랴", "싸", "죽", "뿐"};

// 가중치 파일에서 가중치 읽기 함수
void load_updated_weights(const char *weights_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *num_layers, int *layer_sizes)
{
    FILE *file = fopen(weights_file, "r");
    if (file == NULL)
    {
        perror("가중치 파일을 열 수 없습니다");
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

// BMP 이미지 파일 읽기 함수
void read_bmp_image_1bit(const char *file_path, double *pixels, int input_size)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        perror("이미지 파일을 열 수 없습니다");
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

// 데이터셋 로드 함수
void load_dataset(const char *csv_file, char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH], int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE], int *total_samples)
{
    FILE *file = fopen(csv_file, "r");
    if (file == NULL)
    {
        perror("CSV 파일을 열 수 없습니다");
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file); // 헤더 건너뛰기

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

// Sigmoid 함수
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// ReLU 함수
double relu(double x)
{
    return (x > 0) ? x : 0;
}

// Tanh 함수
double tanh_activation(double x)
{
    return tanh(x);
}

// Softmax 함수 (벡터에 대해 계산)
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
        a[i] = exp(z[i] - max_z); // 최대값을 빼주어 오버플로우 방지
        sum_exp += a[i];
    }

    // 정규화
    for (int i = 0; i < size; i++)
    {
        a[i] /= sum_exp;
    }
}

// 행렬 곱셈 함수 (matrix * vector)
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

// 순전파 함수
void forward_propagation(double *input_data, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *layer_sizes, int num_layers, double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE], const char *activation_function, const char *output_activation_function)
{
    // 첫 번째 레이어의 활성화는 입력 데이터와 동일
    memcpy(activations[0], input_data, sizeof(double) * layer_sizes[0]);

    // 각 레이어를 순차적으로 계산
    for (int i = 1; i < num_layers; i++)
    {
        double z[MAX_LAYER_SIZE] = {0}; // z 벡터
        double a[MAX_LAYER_SIZE] = {0}; // a 벡터 (활성화된 값)

        // 행렬 곱셈 (weights * activations)
        mat_vec_mult(weights[i - 1], activations[i - 1], z, layer_sizes[i], layer_sizes[i - 1]);

        // 활성화 함수 적용
        if (i == num_layers - 1)
        { // 출력 레이어
            if (strcmp(output_activation_function, "sigmoid") == 0)
            {
                // Sigmoid 함수
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = sigmoid(z[j]);
                }
            }
            else if (strcmp(output_activation_function, "softmax") == 0)
            {
                // Softmax 함수
                softmax(z, a, layer_sizes[i]);
            }
        }
        else
        { // 은닉 레이어
            if (strcmp(activation_function, "sigmoid") == 0)
            {
                // Sigmoid 함수
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = sigmoid(z[j]);
                }
            }
            else if (strcmp(activation_function, "relu") == 0)
            {
                // ReLU 함수
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = relu(z[j]);
                }
            }
            else if (strcmp(activation_function, "tanh") == 0)
            {
                // Tanh 함수
                for (int j = 0; j < layer_sizes[i]; j++)
                {
                    a[j] = tanh_activation(z[j]);
                }
            }
        }

        // 활성화 값 저장
        memcpy(activations[i], a, sizeof(double) * layer_sizes[i]);
    }
}

// 모델 평가 함수
void evaluate_model(const char *weights_file, const char *csv_file, int input_size, const char *output_log_file, const char *prediction_log_file)
{
    double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    int layer_sizes[MAX_LAYER_SIZE];
    int num_layers;

    // 가중치 파일에서 가중치 읽기
    load_updated_weights(weights_file, weights, &num_layers, layer_sizes);

    // 데이터셋 로드
    char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH];
    int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE];
    int total_samples;
    load_dataset(csv_file, dataset, labels, &total_samples);

    FILE *label_log = fopen(output_log_file, "w");
    if (label_log == NULL)
    {
        perror("라벨 비교 로그 파일을 열 수 없습니다");
        exit(1);
    }

    FILE *prediction_log = fopen(prediction_log_file, "w");
    if (prediction_log == NULL)
    {
        perror("예측 결과 로그 파일을 열 수 없습니다");
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

        // 예측값 및 실제 라벨 비교 정보 저장
        fprintf(label_log, "이미지 경로: %s\n", dataset[sample_index]);
        fprintf(label_log, "실제 라벨: %s, 목표 인덱스: %d\n", characters_list[target_class], target_class);
        fprintf(label_log, "예측 라벨: %s, 예측 인덱스: %d\n", characters_list[predicted], predicted);
        // 실제 배열과 예측 배열 저장
        fprintf(label_log, "실제 배열: ");
        for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
        {
            fprintf(label_log, "%d ", labels[sample_index][i]); // 실제 라벨 (원핫 인코딩)
        }
        fprintf(label_log, "\n");

        fprintf(label_log, "예측 배열: ");
        for (int i = 0; i < layer_sizes[num_layers - 1]; i++)
        {
            fprintf(label_log, "%.4f ", activations[num_layers - 1][i]); // 예측 확률
        }
        fprintf(label_log, "\n");
        fprintf(label_log, "====================\n");

        if (predicted == target_class)
        {
            fprintf(prediction_log, "실제: %s, 예측: %s - 예측이 맞았습니다.\n", characters_list[target_class], characters_list[predicted]);
            correct_predictions++;
        }
        else
        {
            fprintf(prediction_log, "실제: %s, 예측: %s - 예측이 틀렸습니다.\n", characters_list[target_class], characters_list[predicted]);
        }
    }

    fclose(label_log);
    fclose(prediction_log);

    // 평가 정확도 출력
    double accuracy = (double)correct_predictions / total_samples * 100;
    printf("\n평가 정확도: %.6f%%\n", accuracy);
}

int main()
{
    const char *weights_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/all_training_results_plz/processed_weights_with_labels.txt";
    const char *csv_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/dataset_labels_test.csv";
    const char *output_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/label_comparison_output_all_c.txt";
    const char *prediction_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/prediction_results_c.txt";
    int input_size = 64 * 64;

    evaluate_model(weights_file, csv_file, input_size, output_log_file, prediction_log_file);

    return 0;
}
