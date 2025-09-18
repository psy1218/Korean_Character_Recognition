#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>  // isspace() 함수를 사용하기 위한 헤더 파일

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

void load_updated_weights(const char *weights_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *num_layers, int *layer_sizes, const char *debug_log_file) {
    FILE *file = fopen(weights_file, "r");
    if (file == NULL) {
        perror("가중치 파일을 열 수 없습니다");
        exit(1);
    }

    FILE *debug_log = fopen(debug_log_file, "w");
    if (debug_log == NULL) {
        perror("디버그 로그 파일을 열 수 없습니다");
        fclose(file);
        exit(1);
    }

    char line[4096];
    int layer_index = -1;
    int weight_index = 0;

    while (fgets(line, sizeof(line), file)) {
        // 개행 문자 제거
        line[strcspn(line, "\n")] = '\0';

        // 레이어 정보를 읽는 부분
        if (strncmp(line, "Layer", 5) == 0) {
            layer_index++;
            weight_index = 0;  // 새로운 레이어가 시작되면 인덱스를 0으로 초기화

            // "Layer 1 Weights:" 형식의 문자열에서 레이어 번호를 파싱합니다.
            int temp_layer_index;
            if (sscanf(line, "Layer %d Weights:", &temp_layer_index) == 1) {
                // 레이어 크기를 파싱하거나 기본값 설정
                if (sscanf(line, "Layer %*d Weights: %d", &layer_sizes[layer_index]) != 1) {
                    // 만약 레이어 크기가 파일에 명시되지 않은 경우, 기본값 사용
                    layer_sizes[layer_index] = MAX_LAYER_SIZE;  // 기본값 설정 (필요 시 수정)
                }
            } else {
                fprintf(stderr, "레이어 정보를 읽는 데 실패했습니다: %s\n", line);
                fclose(file);
                fclose(debug_log);
                exit(1);
            }

            *num_layers = layer_index + 1;
        } else {
            // 숫자 문자열을 파싱하기 위한 포인터
            char *token = line;
            while (*token) {
                // 공백을 건너뜀
                while (*token && isspace(*token)) {
                    token++;
                }

                // 숫자 읽기
                double value;
                if (sscanf(token, "%lf", &value) == 1) {
                    // 현재 레이어의 크기를 초과하는지 확인
                    if (weight_index >= layer_sizes[layer_index]) {
                        fprintf(stderr, "가중치 인덱스 초과: Layer %d, Weight %d\n", layer_index, weight_index);
                        fclose(file);
                        fclose(debug_log);
                        exit(1);
                    }

                    weights[layer_index][weight_index++] = value;
                    fprintf(debug_log, "Layer %d, Weight %d: %f\n", layer_index, weight_index - 1, value);

                    // 다음 숫자로 이동
                    while (*token && !isspace(*token)) {
                        token++;
                    }
                } else {
                    break;  // 숫자가 아닌 경우 루프 탈출
                }
            }
        }
    }

    fclose(file);
    fclose(debug_log);
}

// 가중치 저장 함수 (파일로 저장)
void save_weights(const char *output_file, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int num_layers, int *layer_sizes) {
    FILE *file = fopen(output_file, "w");
    if (file == NULL) {
        perror("가중치 저장 파일을 열 수 없습니다");
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

// 데이터셋 로드 함수
void load_dataset(const char *csv_file, char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH], int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE], int *total_samples) {
    FILE *file = fopen(csv_file, "r");
    if (file == NULL) {
        perror("CSV 파일을 열 수 없습니다");
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file); // 헤더 건너뛰기

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

// BMP 이미지 파일 읽기 함수
void read_bmp_image_1bit(const char *file_path, double *pixels, int input_size) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("이미지 파일을 열 수 없습니다");
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

// 순전파 함수
void forward_propagation(double *input_data, double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE], int *layer_sizes, int num_layers, double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE], const char *activation_function, const char *output_activation_function) {
    // 첫 번째 레이어의 활성화는 입력 데이터와 동일
    memcpy(activations[0], input_data, sizeof(double) * layer_sizes[0]);

    // 각 레이어를 순차적으로 계산
    for (int i = 1; i < num_layers; i++) {
        double z[MAX_LAYER_SIZE] = {0}; // z 벡터
        double a[MAX_LAYER_SIZE] = {0}; // a 벡터 (활성화된 값)

        // 행렬 곱셈 (weights * activations)
        for (int j = 0; j < layer_sizes[i]; j++) {
            z[j] = 0;
            for (int k = 0; k < layer_sizes[i - 1]; k++) {
                z[j] += weights[i - 1][j * layer_sizes[i - 1] + k] * activations[i - 1][k];
            }
        }

        // 활성화 함수 적용
        if (i == num_layers - 1) { // 출력 레이어
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
        } else { // 은닉 레이어
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

        // 활성화 값 저장
        memcpy(activations[i], a, sizeof(double) * layer_sizes[i]);
    }
}

// 모델 평가 함수
void evaluate_model(const char *weights_file, const char *csv_file, int input_size, const char *output_log_file, const char *prediction_log_file, const char *forward_log_file, const char *weights_output_file, const char *debug_log_file) {
    double weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    int layer_sizes[MAX_LAYER_SIZE];
    int num_layers;

    // 가중치 파일에서 가중치 읽기
    load_updated_weights(weights_file, weights, &num_layers, layer_sizes, debug_log_file);

    // 가중치 저장하기 (가중치 확인용)
    save_weights(weights_output_file, weights, num_layers, layer_sizes);

    // 데이터셋 로드
    char dataset[MAX_LAYER_SIZE][MAX_PATH_LENGTH];
    int labels[MAX_LAYER_SIZE][MAX_CHAR_LIST_SIZE];
    int total_samples;
    load_dataset(csv_file, dataset, labels, &total_samples);

    FILE *label_log = fopen(output_log_file, "w");
    if (label_log == NULL) {
        perror("라벨 비교 로그 파일을 열 수 없습니다");
        exit(1);
    }

    FILE *prediction_log = fopen(prediction_log_file, "w");
    if (prediction_log == NULL) {
        perror("예측 결과 로그 파일을 열 수 없습니다");
        exit(1);
    }

    // 순전파 로그 파일 열기
    FILE *forward_log = fopen(forward_log_file, "w");
    if (forward_log == NULL) {
        perror("순전파 로그 파일을 열 수 없습니다");
        exit(1);
    }

    int correct_predictions = 0;
    double activations[MAX_LAYER_SIZE][MAX_LAYER_SIZE];

    for (int sample_index = 0; sample_index < total_samples; sample_index++) {
        double input_data[MAX_IMAGE_SIZE];
        read_bmp_image_1bit(dataset[sample_index], input_data, input_size);

        // 순전파 계산
        fprintf(forward_log, "=== Sample %d ===\n", sample_index + 1);
        fprintf(forward_log, "이미지 경로: %s\n", dataset[sample_index]);
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

    // 평가 정확도 출력
    double accuracy = (double)correct_predictions / total_samples * 100;
    printf("\n평가 정확도: %.6f%%\n", accuracy);
}

int main() {
    const char
