import os
import csv
import random
import time
import numpy as np
import threading

batch_info = {"current_batch": 0, "current_epoch": 0, "batch_size": 0, "total_batches": 0}


# 활성화 함수 (Sigmoid, ReLU, Tanh, Softmax 추가)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.power(x, 2)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Overflow 방지를 위해 x에서 최대값을 빼줌
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


# 가중치 초기화 함수 (Xavier, He 초기화 추가)
def initialize_weights(rows, cols, method='xavier', seed=None):
    if seed is not None:
        np.random.seed(seed)

    if method == 'xavier':
        limit = np.sqrt(6 / (rows + cols))
        return np.random.uniform(-limit, limit, (rows, cols))
    elif method == 'he':
        limit = np.sqrt(2 / rows)
        return np.random.uniform(-limit, limit, (rows, cols))
    else:
        return np.random.uniform(-1, 1, (rows, cols))


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

    return np.array(pixels, dtype=np.float32).reshape(-1, 1)


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
                label = np.array([float(v) for v in row[1:]], dtype=np.float32).reshape(-1, 1)
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


# 순전파 함수 (수정됨: 출력층 활성화 함수 추가)
def forward_propagation(input_data, weights, activation_function, output_activation_function):
    activations = [input_data]

    for i in range(len(weights)):
        z = np.dot(weights[i], activations[-1])
        if i == len(weights) - 1:  # 수정됨: 출력층의 활성화 함수 적용
            if output_activation_function == 'sigmoid':
                a = sigmoid(z)
            elif output_activation_function == 'softmax':
                a = softmax(z)
        else:
            if activation_function == 'sigmoid':
                a = sigmoid(z)
            elif activation_function == 'relu':
                a = relu(z)
            elif activation_function == 'tanh':
                a = tanh(z)
        activations.append(a)

    return activations


# 역전파 함수
def backpropagation(weights, activations, target, learning_rate, activation_function):
    output = activations[-1]
    output_error = target - output
    mse = np.mean(np.square(output_error))

    if activation_function == 'sigmoid':
        output_delta = output_error * sigmoid_derivative(output)
    elif activation_function == 'relu':
        output_delta = output_error * relu_derivative(output)
    elif activation_function == 'tanh':
        output_delta = output_error * tanh_derivative(output)

    deltas = [output_delta]
    for i in range(len(weights) - 1, 0, -1):
        layer_error = np.dot(weights[i].T, deltas[-1])
        if activation_function == 'sigmoid':
            layer_delta = layer_error * sigmoid_derivative(activations[i])
        elif activation_function == 'relu':
            layer_delta = layer_error * relu_derivative(activations[i])
        elif activation_function == 'tanh':
            layer_delta = layer_error * tanh_derivative(activations[i])
        deltas.append(layer_delta)

    deltas.reverse()

    for i in range(len(weights)):
        weights[i] += learning_rate * np.dot(deltas[i], activations[i].T)

    return output_error, mse


# 정확도 계산 함수
def calculate_accuracy(weights, inputs, targets, activation_function, output_activation_function):
    if len(targets) == 0:
        return 0

    correct_predictions = 0
    for input_data, target in zip(inputs, targets):
        activations = forward_propagation(input_data, weights, activation_function, output_activation_function)
        output = activations[-1]
        predicted = (output > 0.5).astype(int) if output_activation_function == 'sigmoid' else np.argmax(output, axis=0)
        target_class = target if output_activation_function == 'sigmoid' else np.argmax(target, axis=0)
        if np.array_equal(predicted, target_class):
            correct_predictions += 1
    return correct_predictions / len(targets) * 100


# 사용자 입력을 감지하는 함수 (스레드로 실행)
def listen_for_keypress(start_time):
    while True:
        input_key = input()
        if input_key:
            elapsed_time = time.time() - start_time
            remaining_batches = (batch_info["total_batches"] - batch_info["current_batch"] // batch_info["batch_size"])
            print(f"경과 시간: {elapsed_time:.2f}초")
            print(f"현재 에포크: {batch_info['current_epoch']} / 현재 배치: {batch_info['current_batch']} / 남은 배치: {remaining_batches}")
            print(f"훈련 정확도: {float(batch_info.get('train_accuracy', 0.0)):.6f}% / 테스트 정확도: {float(batch_info.get('test_accuracy', 0.0)):.6f}%")
            print(f"현재 MSE: {float(batch_info.get('mse', 0.0)):.6f}")

# 메인 실행 함수
def main():
    start_time = time.time()

    threading.Thread(target=listen_for_keypress, args=(start_time,), daemon=True).start()  # 사용자 입력 감지 스레드 시작
    
    last_log_time = start_time  # 마지막 로그 시간 초기화

    input_size = 64 * 64
    base_path = "C:/Users/asx12/OneDrive/바탕 화면/인공지능"
    csv_file = "dataset_labels.csv"
    batch_size = 64
    learning_rate = 0.001
    epochs = 200
    activation_function = 'relu'  # 활성화 함수 선택 ('sigmoid', 'relu', 'tanh' 중 선택)
    output_activation_function = 'softmax'  # 출력층 활성화 함수 설정 ('sigmoid', 'softmax' 중 선택) - 수정됨
    weight_init_method = 'he'  # 가중치 초기화 방법 선택 ('xavier', 'he' 중 선택)

    results_folder = "all_training_results_plz_size"
    os.makedirs(results_folder, exist_ok=True)

    try:
        inputs, targets = load_dataset(base_path, csv_file, input_size)
    except ValueError as e:
        print(e)
        return

    # 데이터셋을 무작위로 섞음
    combined = list(zip(inputs, targets))
    random.shuffle(combined)
    inputs, targets = zip(*combined)

    train_size = int(0.8 * len(inputs))
    train_inputs, train_targets = inputs[:train_size], targets[:train_size]
    test_inputs, test_targets = inputs[train_size:], targets[train_size:]

    output_size = len(train_targets[0])  # 출력 노드 수를 레이블의 길이로 설정
    layer_structure = [input_size, 512, 256, 126, output_size]  # 마지막 레이어 노드 수를 출력 크기로 설정
    weights = [initialize_weights(layer_structure[i + 1], layer_structure[i], method=weight_init_method, seed=42) for i in range(len(layer_structure) - 1)]

    # Update batch information - 수정됨: 배치 크기 및 전체 배치 수 정보 저장
    batch_info["batch_size"] = batch_size
    batch_info["total_batches"] = len(train_inputs) // batch_size

    try:
        best_mse = float('inf')  # 초기 최적의 MSE 설정
        best_weights = []
        for epoch in range(epochs):
            batch_info["current_epoch"] = epoch + 1

            # 각 에포크마다 데이터셋 무작위로 섞기 - 수정됨
            combined = list(zip(train_inputs, train_targets))
            random.shuffle(combined)
            train_inputs, train_targets = zip(*combined)

            # 각 에포크마다 결과 저장 폴더 생성
            epoch_folder = os.path.join(results_folder, f"epoch_{epoch + 1}")
            os.makedirs(epoch_folder, exist_ok=True)

            for batch_start in range(0, len(train_inputs), batch_size):
                batch_info["current_batch"] = batch_start

                batch_inputs = train_inputs[batch_start:batch_start + batch_size]
                batch_targets = train_targets[batch_start:batch_start + batch_size]

                for input_data, target in zip(batch_inputs, batch_targets):
                    activations = forward_propagation(input_data, weights, activation_function, output_activation_function)  # 수정됨: 출력층 활성화 함수 추가
                    output_error, mse = backpropagation(weights, activations, target, learning_rate, activation_function)

                if mse < best_mse:
                    best_mse = mse
                    best_weights = [layer.copy() for layer in weights]

            train_accuracy = calculate_accuracy(weights, train_inputs, train_targets, activation_function, output_activation_function)
            test_accuracy = calculate_accuracy(weights, test_inputs, test_targets, activation_function, output_activation_function)
            
            with open(os.path.join(epoch_folder, f"epoch_{epoch + 1}_results.txt"), "w", encoding="utf-8") as f:
                f.write(f"에포크 {epoch + 1} 결과\n")
                f.write(f"====================\n")
                f.write(f"훈련 정확도: {train_accuracy:.6f}%\n")
                f.write(f"테스트 정확도: {test_accuracy:.6f}%\n")
                
            # 에포크마다 각 레이어의 출력값, 목표값, 오차 저장
            with open(os.path.join(epoch_folder, f"epoch_{epoch + 1}_details.txt"), "a", encoding="utf-8") as f:               
                f.write(f"\n--- MSE (Mean Squared Error) ---\n")
                f.write(f"{round(best_mse, 4)}\n")
                
                f.write(f"Target: {target.flatten()}\n")
                f.write(f"\n--- Output Error ---\n")
                f.write(f"{[round(e, 4) for e in output_error.flatten()]}\n")
                f.write(f"Input Data: {input_data.flatten()}\n")

                for layer_index, activation in enumerate(activations):
                    f.write(f"\n--- Layer {layer_index + 1} Activation ---\n")
                    f.write(f"{[round(a[0], 4) for a in activation]}\n")
                
                f.write(f"\n--- Updated Weights ---\n")
                for layer_index, layer_weights in enumerate(best_weights):
                    f.write(f"\n--- Layer {layer_index + 1} Weights ---\n")
                    for weight_row in layer_weights:
                        f.write(f"{[round(w, 4) for w in weight_row]}\n")

            # 1시간마다 학습 상태 출력
            current_time = time.time()
            if current_time - last_log_time >= 3600:
                elapsed_time = current_time - start_time
                print(f"학습 중입니다. 경과 시간: {elapsed_time:.2f}초")
                print(f"현재 에포크: {batch_info['current_epoch']} / 현재 배치: {batch_info['current_batch']}")
                print(f"훈련 정확도: {train_accuracy:.6f}% / 테스트 정확도: {test_accuracy:.6f}%")
                print(f"현재 MSE: {mse:.6f}")
                last_log_time = current_time

    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다!")

    finally:
        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time
        print(f"경과 시간: {elapsed_time:.2f}초")

if __name__ == "__main__":
    main()
