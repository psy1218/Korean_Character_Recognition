# Sigmoid 활성화 함수와 그 도함수
# x값을 입력으로 받아 Sigmoid 활성화 함수를 적용하여 결과를 반환
# Sigmoid 함수는 신경망의 출력값을 0과 1 사이로 매핑

def sigmoid(x):
    return 1 / (1 + (2.718281828459045 ** -x))

# Sigmoid 함수의 도함수
# x값을 입력으로 받아 Sigmoid 도함수의 결과를 반환
# 가중치 업데이트 시 기울기 계산에 사용

def sigmoid_derivative(x):
    return x * (1 - x)

# 파일에서 가중치 불러오기
# 주어진 파일에서 가중치를 읽어와 각 레이어별로 나누어 리스트 형태로 반환

def load_weights(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    weights = []
    layer_idx = -1
    for line in lines:
        if line.startswith("--- Layer"):
            # 레이어 시작 부분을 만날 때마다 새로운 레이어 추가
            layer_idx += 1
            weights.append([])
        elif line.startswith("["):
            # 가중치 값을 리스트 형태로 저장
            weights[layer_idx].append([float(x) for x in line.strip()[1:-1].split(", ")])
    return weights

# 파일에 가중치 저장하기
# 가중치를 주어진 파일에 레이어별로 저장

def save_weights(filename, weights):
    with open(filename, "w", encoding="utf-8") as file:
        for i, layer_weights in enumerate(weights):
            file.write(f"--- Layer {i + 1} 가중치 ---\n")
            for neuron_weights in layer_weights:
                file.write(str(neuron_weights) + "\n")

# 역전파 함수
# 가중치를 업데이트하기 위해 역전파 알고리즘을 사용하여 오차와 델타를 계산
# 업데이트된 가중치를 새로운 파일에 저장

def backpropagation(weights_filename, output_filename, learning_rate=0.01):
    # 가중치 불러오기
    weights = load_weights(weights_filename)
    output_log = []
    
    # 출력 파일 읽기
    with open(output_filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # 레이어별 출력을 로그에서 추출
        layer_logs = [line.strip() for line in lines if line.startswith("Layer")]
        output_values = [float(x.strip().strip(']')) for x in lines[-3].split(": ")[1].strip("[]").split(", ")]
        target_values = [float(x.strip().strip(']')) for x in lines[-2].split(": ")[1].strip("[]").split(", ")]
        error = float(lines[-1].split(": ")[1])
        
        # 각 레이어의 출력을 리스트로 저장
        for i in range(len(layer_logs)):
            layer_output = [float(x.strip().strip(']')) for x in layer_logs[i].split(": ")[1].strip("[]").split(", ")]
            output_log.append(layer_output)

    # 출력층 오차와 델타 계산
    # 출력값과 목표값 간의 차이를 계산하여 출력층 오차를 구함
    output_layers = output_log[-1]
    targets = target_values
    output_error = [targets[i] - output_layers[i] for i in range(len(targets))]
    # 출력 오차에 Sigmoid 도함수를 곱하여 델타 계산
    output_delta = [output_error[i] * sigmoid_derivative(output_layers[i]) for i in range(len(output_layers))]

    deltas = [output_delta]
    layer_errors = [output_error]

    # 각 레이어의 델타 계산
    # 역전파를 통해 각 레이어에 대한 오차와 델타 값을 계산
    for i in range(len(weights) - 2, -1, -1):
        error = [0] * len(output_log[i])
        for j in range(len(weights[i + 1])):
            for k in range(len(weights[i])):
                if j < len(deltas[-1]) and k < len(weights[i + 1][j]):
                    error[k] += deltas[-1][j] * weights[i + 1][j][k]
        layer_errors.append(error)
        delta = [error[j] * sigmoid_derivative(output_log[i][j]) for j in range(len(output_log[i]))]
        deltas.append(delta)

    # 델타 리스트를 순서를 맞추기 위해 뒤집기
    deltas.reverse()
    
    # 가중치 업데이트
    # 각 레이어별로 학습률과 델타를 반영하여 가중치 업데이트
    for i in range(len(weights)):
        layer_input = output_log[i - 1] if i > 0 else output_log[i]
        for j in range(len(weights[i])):
            if isinstance(weights[i][j], list):
                for k in range(len(weights[i][j])):
                    if k < len(layer_input) and j < len(deltas[i]):
                        weights[i][j][k] += learning_rate * layer_input[k] * deltas[i][j]

    # 업데이트된 가중치 저장
    save_weights("updated_weights.txt", weights)

# 사용 예시
# weights_log.txt와 output_log.txt 파일을 사용하여 가중치를 업데이트하고 updated_weights.txt에 저장
backpropagation("weights_log.txt", "output_log.txt", learning_rate=0.01)
