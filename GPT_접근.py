import os
import csv
import random
import time
import math

# 활성화 함수 (ReLU, Sigmoid, Tanh)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

# 행렬 곱 함수
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("행렬 A의 열 수와 B의 행 수가 일치해야 합니다.")

    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# 가중치 초기화 함수 (Xavier, He 초기화 추가)
def initialize_weights(rows, cols, method='xavier', seed=None):
    if seed is not None:
        random.seed(seed)

    if method == 'xavier':
        limit = math.sqrt(6 / (rows + cols))
    elif method == 'he':
        limit = math.sqrt(2 / rows)
    else:
        limit = 1

    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

# BMP 이미지 파일 읽기 함수
def read_bmp_image_1bit(file_path, input_size):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "rb") as f:
        f.seek(54)  # BMP 헤더 건너뛰기
        pixel_data = f.read()

    pixels = []
    for byte in pixel_data:
        for i in range(8):
            pixels.append((byte >> (7 - i)) & 1)

    if len(pixels) > input_size:
        pixels = pixels[:input_size]
    elif len(pixels) < input_size:
        raise ValueError("이미지 크기가 입력 데이터 크기보다 작습니다.")

    return [[float(p)] for p in pixels]

# 데이터셋 로드 함수
def load_dataset(base_path, csv_file, input_size):
    image_files, labels = [], []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            try:
                file_path = os.path.join(base_path, row[0].strip())
                image = read_bmp_image_1bit(file_path, input_size)
                label = [float(v) for v in row[1:]]
                image_files.append(image)
                labels.append(label)
            except ValueError as e:
                print(f"잘못된 라벨 형식 발견: {row}. 에러: {e}")
                continue
            except FileNotFoundError as e:
                print(e)
                continue

    if not image_files or not labels:
        raise ValueError("데이터셋 로드에 실패했습니다. 이미지 파일이나 라벨이 없습니다.")

    return image_files, labels

# 순전파 함수
def forward_propagation(input_data, weights, activation_function):
    activations = [input_data]

    for i in range(len(weights)):
        z = matrix_multiply(weights[i], activations[-1])
        if activation_function == 'sigmoid':
            a = [[sigmoid(val[0])] for val in z]
        elif activation_function == 'relu':
            a = [[relu(val[0])] for val in z]
        elif activation_function == 'tanh':
            a = [[tanh(val[0])] for val in z]
        activations.append(a)

    return activations

# 역전파 함수
def backpropagation(weights, activations, target, learning_rate, activation_function):
    output = activations[-1]
    output_error = [target[i] - output[i][0] for i in range(len(target))]
    mse = sum((target[i] - output[i][0]) ** 2 for i in range(len(target))) / len(target)

    if activation_function == 'sigmoid':
        output_delta = [output_error[i] * sigmoid_derivative(output[i][0]) for i in range(len(output))]
    elif activation_function == 'relu':
        output_delta = [output_error[i] * relu_derivative(output[i][0]) for i in range(len(output))]
    elif activation_function == 'tanh':
        output_delta = [output_error[i] * tanh_derivative(output[i][0]) for i in range(len(output))]

    deltas = [output_delta]
    for i in range(len(weights) - 1, 0, -1):
        layer_error = [0] * len(activations[i])
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                layer_error[k] += deltas[-1][j] * weights[i][j][k]
        if activation_function == 'sigmoid':
            layer_delta = [layer_error[j] * sigmoid_derivative(activations[i][j][0]) for j in range(len(layer_error))]
        elif activation_function == 'relu':
            layer_delta = [layer_error[j] * relu_derivative(activations[i][j][0]) for j in range(len(layer_error))]
        elif activation_function == 'tanh':
            layer_delta = [layer_error[j] * tanh_derivative(activations[i][j][0]) for j in range(len(layer_error))]
        deltas.append(layer_delta)

    deltas.reverse()

    for i in range(len(weights)):
        layer_input = activations[i]
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] += learning_rate * deltas[i][j] * layer_input[k][0]

    return output_error, mse

# 정확도 계산 함수
def calculate_accuracy(weights, inputs, targets, activation_function):
    if len(targets) == 0:
        return 0

    correct_predictions = 0
    for input_data, target in zip(inputs, targets):
        activations = forward_propagation(input_data, weights, activation_function)
        output = activations[-1]
        predicted = [1 if o[0] > 0.5 else 0 for o in output]
        if predicted == target:
            correct_predictions += 1
    return correct_predictions / len(targets) * 100

# 메인 실행 함수
def main():
    start_time = time.time()  # 시작 시간 기록

    input_size = 64 * 64
    base_path = "C:/Users/asx12/OneDrive/바탕 화면/인공지능"  # 이미지 폴더 경로
    csv_file = "dataset_labels.csv"
    batch_size = 10
    learning_rate = 0.01
    epochs = 10
    activation_function = 'relu'  # 'sigmoid', 'relu', 'tanh' 중 하나 선택
    weight_init_method = 'he'  # 'xavier', 'he' 중 하나 선택

    results_folder = "training_results3"
    os.makedirs(results_folder, exist_ok=True)

    try:
        inputs, targets = load_dataset(base_path, csv_file, input_size)
    except ValueError as e:
        print(e)
        return

    train_size = int(0.8 * len(inputs))
    train_inputs, train_targets = inputs[:train_size], targets[:train_size]
    test_inputs, test_targets = inputs[train_size:], targets[train_size:]

    layer_structure = [input_size, 128, 64, 4]
    weights = [initialize_weights(layer_structure[i + 1], layer_structure[i], method=weight_init_method, seed=42) for i in range(len(layer_structure) - 1)]

    for epoch in range(epochs):
        epoch_folder = os.path.join(results_folder, f"epoch_{epoch + 1}")
        os.makedirs(epoch_folder, exist_ok=True)

        for batch_start in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[batch_start:batch_start + batch_size]
            batch_targets = train_targets[batch_start:batch_start + batch_size]

            for input_data, target in zip(batch_inputs, batch_targets):
                activations = forward_propagation(input_data, weights, activation_function)
                output_error, mse = backpropagation(weights, activations, target, learning_rate, activation_function)

                # 각 레이어의 출력값, 목표값, 오차 저장
                with open(os.path.join(epoch_folder, f"batch_{batch_start}_details.txt"), "a", encoding="utf-8") as f:
                    f.write(f"Input Data: {input_data}\n")
                    f.write(f"Target: {target}\n")
                    for layer_index, activation in enumerate(activations):
                        f.write(f"\n--- Layer {layer_index + 1} Activation ---\n")
                        f.write(f"{[round(a[0], 4) for a in activation]}\n")
                    f.write(f"\n--- Output Error ---\n")
                    f.write(f"{[round(e, 4) for e in output_error]}\n")
                    f.write(f"\n--- MSE (Mean Squared Error) ---\n")
                    f.write(f"{round(mse, 4)}\n")
                    f.write(f"\n--- Updated Weights ---\n")
                    for layer_index, layer_weights in enumerate(weights):
                        f.write(f"\n--- Layer {layer_index + 1} Weights ---\n")
                        for weight_row in layer_weights:
                            f.write(f"{[round(w, 4) for w in weight_row]}\n")

        train_accuracy = calculate_accuracy(weights, train_inputs, train_targets, activation_function)
        test_accuracy = calculate_accuracy(weights, test_inputs, test_targets, activation_function)

        with open(os.path.join(epoch_folder, f"epoch_{epoch + 1}_results.txt"), "w", encoding="utf-8") as f:
            f.write(f"에포크 {epoch + 1} 결과\n")
            f.write(f"====================\n")
            f.write(f"훈련 정확도: {train_accuracy:.2f}%\n")
            f.write(f"테스트 정확도: {test_accuracy:.2f}%\n")

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time
    print(f"학습 완료. 결과가 저장되었습니다. 경과 시간: {elapsed_time:.2f}초")

if __name__ == "__main__":
    main()
