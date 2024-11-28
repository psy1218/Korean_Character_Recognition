import os
import csv
import numpy as np
import time
from PIL import Image

# 원핫 인코딩한 순서에 따른 글자 리스트
characters_list = ['충', '청', '남', '도', '홍', '성', '군', '읍', '기', '길', '숙', '사', '호', '동', '박', '소', '윤', '김', '은', '옥', '영', '수', '종', '헌', '조', '인', '의', '경', '봉', '화', '천', '광', '역', '시', '연', '구', '아', '카', '데', '미', '로', '춘', '이', '는', '듣', '만', '하', '여', '가', '슴', '설', '레', '말', '다', '너', '두', '손', '을', '에', '대', '고', '물', '방', '같', '심', '장', '들', '어', '보', '라', '피', '끓', '뛰', '노', '거', '선', '관', '힘', '있', '것', '류', '를', '꾸', '며', '내', '려', '온', '력', '바', '투', '명', '되', '얼', '음', '과', '으', '지', '혜', '날', '우', '나', '갑', '속', '든', '칼', '니', '더', '면', '간', '마', '쓸', '랴', '싸', '죽', '뿐']

# 가중치 파일에서 가중치 읽기 함수
def load_updated_weights(weights_file):
    weights = []
    with open(weights_file, 'r') as f:
        lines = f.readlines()
        layer_weights = []
        recording = False
        with open("C:/Users/asx12/OneDrive/바탕 화면/인공지능/loaded_weights_output.txt", 'w', encoding="utf-8") as log_file:
            for line in lines:
                if line.startswith('Layer') and 'Weights' in line:
                    # 새로운 레이어가 시작될 때 이전 레이어 가중치를 저장
                    if layer_weights:
                        weights.append(np.array(layer_weights, dtype=float))
                        for weight_row in layer_weights:
                            log_file.write(' '.join(map(str, weight_row)) + '\n')
                        layer_weights = []
                    recording = True
                elif recording:
                    # 가중치 값을 읽어와서 float 리스트로 변환 후 저장
                    layer_weights.append([float(w) for w in line.strip().split()])
            
            # 마지막 레이어의 가중치 추가
            if layer_weights:
                weights.append(np.array(layer_weights, dtype=float))
                for weight_row in layer_weights:
                    log_file.write(' '.join(map(str, weight_row)) + '\n')

    return weights

# 학습 중 최종 저장된 가중치(best_weights)와 저장된 가중치 비교 함수
def compare_weights(loaded_weights, trained_weights):
    if len(loaded_weights) != len(trained_weights):
        print("레이어 개수가 다릅니다.")
        return False

    for i in range(len(loaded_weights)):
        if not np.allclose(loaded_weights[i], trained_weights[i], atol=1e-5):
            print(f"레이어 {i + 1}의 가중치가 다릅니다.")
            return False
    print("가중치가 모두 일치합니다.")
    return True

# 가중치 파일에서 읽은 내용을 별도의 파일로 저장하는 함수
def save_loaded_weights(weights, output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        for i, layer_weights in enumerate(weights):
            f.write(f'Layer {i + 1} Weights:\n')
            for weight_row in layer_weights:
                f.write(' '.join(map(str, weight_row)) + '\n')
            f.write('\n')

# BMP 이미지 파일 읽기 함수
def read_bmp_image_1bit(file_path, input_size):
    with open(file_path, "rb") as f:
        f.seek(54)
        pixel_data = f.read()
    
    pixels = []
    for byte in pixel_data:
        for i in range(8):
            pixels.append((byte >> (7 - i)) & 1)
    
    if len(pixels) > input_size:
        pixels = pixels[:input_size]
    elif len(pixels) < input_size:
        raise ValueError("이미지 크기가 입력 데이터 크기보다 작습니다.")

    return np.array(pixels, dtype=np.float32).reshape(-1, 1)

# 데이터셋 로드 함수
def load_dataset(base_path, csv_file):
    dataset = {}
    with open(csv_file, newline="", encoding="euc-kr") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header if present
        for row in reader:
            image_path = row[0]
            label = [int(v) for v in row[1:]]
            dataset[image_path] = label
    return dataset

# 순전파 함수
def forward_propagation(input_data, weights, activation_function, output_activation_function):
    activations = [input_data]

    for i in range(len(weights)):
        z = np.dot(weights[i], activations[-1])
        if i == len(weights) - 1:  # Output layer activation
            if output_activation_function == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
            elif output_activation_function == 'softmax':
                exp_z = np.exp(z - np.max(z))  # Overflow 방지를 위해 z에서 최대값을 빼줌
                a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            if activation_function == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
            elif activation_function == 'relu':
                a = np.maximum(0, z)
            elif activation_function == 'tanh':
                a = np.tanh(z)
        activations.append(a)

    return activations

# 모델 평가 함수
def evaluate_model(weights_file, base_path, csv_file, input_size, output_log_file, prediction_log_file, activation_function='relu', output_activation_function='softmax'):
    # 가중치 읽기
    weights = load_updated_weights(weights_file)

    # 데이터셋 로드 및 전체 데이터 평가
    dataset = load_dataset(base_path, csv_file)
    total_samples = len(dataset)
    correct_predictions = 0

    with open(output_log_file, 'w', encoding="utf-8") as label_log, open(prediction_log_file, 'w', encoding="utf-8") as prediction_log:
        for sample, target in dataset.items():
            # 입력 데이터 및 타겟 가져오기
            input_data = read_bmp_image_1bit(os.path.join(base_path, sample), input_size)
            
            # 순전파 수행
            activations = forward_propagation(input_data, weights, activation_function, output_activation_function)
            output = activations[-1]
            
            # 예측값 및 실제 라벨 비교
            predicted = np.argmax(output, axis=0)
            target_class = np.argmax(target)

            character_name = os.path.basename(os.path.dirname(sample))
            predicted_character = characters_list[predicted[0]]  # 원핫인코딩 순서에 따라 예측 라벨 적용

            # 예측 벡터에서 가장 높은 값과 그 위치 찾기
            max_value = np.max(output)
            max_index = np.argmax(output)

            # 라벨 비교 정보 저장
            label_log.write(f"이미지 경로: {sample}\n")
            label_log.write(f"실제 라벨: {character_name}, 목표 인덱스: {target_class}\n")
            label_log.write(f"예측 라벨: {predicted_character}, 예측 인덱스: {predicted[0]}\n")
            label_log.write(f"타겟 벡터: {target}\n")
            label_log.write(f"예측 벡터: {output.flatten()}\n")
            label_log.write(f"예측 벡터의 최대값: {max_value}, 위치: {max_index}\n")
            label_log.write("====================\n")

            if predicted_character == character_name:
                prediction_log.write(f"실제: {character_name}, 예측: {predicted_character} - 예측이 맞았습니다.\n")
                correct_predictions += 1
            else:
                prediction_log.write(f"실제: {character_name}, 예측: {predicted_character} - 예측이 틀렸습니다.\n")

    # 평가 정확도 출력
    accuracy = correct_predictions / total_samples * 100
    print(f"\n평가 정확도: {accuracy:.6f}%")

# 메인 함수
def main():
    weights_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/all_training_results_plz/processed_weights_with_labels.txt"
    base_path = "C:/Users/asx12/OneDrive/바탕 화면/인공지능"
    csv_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/dataset_labels_test.csv"
    input_size = 64 * 64
    output_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/label_comparison_output_all.txt"
    prediction_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/prediction_results.txt"
    
    # 가중치 읽기 및 평가
    evaluate_model(weights_file, base_path, csv_file, input_size, output_log_file, prediction_log_file)
    
    # 가중치 읽기 결과 저장
    output_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/loaded_weights_output.txt"
    weights = load_updated_weights(weights_file)
    save_loaded_weights(weights, output_file)
    print(f"가중치 파일이 '{output_file}'에 저장되었습니다.")

    # 학습된 가중치와 저장된 가중치 비교
    # best_weights는 실제 학습 코드에서 추출해야 합니다.
    # 예시로 best_weights를 가정합니다.
    best_weights = weights  # 여기에 실제 학습된 가중치를 넣으세요.
    comparison_result = compare_weights(weights, best_weights)

if __name__ == "__main__":
    main()
