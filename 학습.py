import os
import random
import csv
import time  # 시간 측정 모듈

# 활성화 함수 (시그모이드)
def sigmoid(x):
    return 1 / (1 + 2.71828 ** -x)

# 활성화 함수 (ReLU)
def relu(x):
    return max(0, x)

# 벡터에 ReLU 적용하기
def relu_vector(matrix):
    return [[relu(val) for val in row] for row in matrix]

# 행렬 초기화 함수
def initialize_matrix(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

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

# 1비트 BMP 이미지를 읽어 1차원 배열로 변환하는 함수
def read_bmp_image_1bit(file_path, input_size):
    with open(file_path, "rb") as f:
        f.seek(54)  # BMP 헤더(54바이트)를 건너뜀
        pixel_data = f.read()

    # 1비트 BMP에서 각 바이트의 비트를 읽어 0 또는 1로 변환
    pixels = []
    for byte in pixel_data:
        for i in range(8):
            pixels.append((byte >> (7 - i)) & 1)  # 각 비트를 추출 (0 또는 1)

    # 입력 데이터 크기에 맞게 자르기
    if len(pixels) > input_size:
        pixels = pixels[:input_size]
    elif len(pixels) < input_size:
        raise ValueError("이미지 크기가 입력 데이터 크기보다 작습니다.")

    # 1차원 배열을 2차원 형태로 반환
    return [[float(p)] for p in pixels]

# 라벨과 이미지 파일 정보를 읽어오는 함수
def load_dataset(base_path, csv_file, input_size):
    image_files = []
    labels = []

    # CSV 파일 읽기
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 첫 번째 행(헤더) 무시

        for row in reader:
            try:
                file_path = os.path.join(base_path, row[0])
                # 문자열 라벨을 숫자 라벨로 변환 (예: "충" -> [1, 0, 0, 0])
                #label = [float(v) for v in row[1:]]
                #image_files.append(read_bmp_image_1bit(file_path, input_size))
                #labels.append(label)
                
                 # BMP 이미지 데이터 읽기
                image = read_bmp_image_1bit(file_path, input_size)
                # 라벨 불러오기
                label = [float(v) for v in row[1:]]
                image_files.append(image)
                labels.append(label)
            except ValueError as e:
                print(f"잘못된 라벨 형식 발견: {row}. 에러: {e}")
                continue  # 잘못된 데이터는 무시

    return image_files, labels

# 순전파 함수
def forward_propagation(layers, weights):
    activations = [layers[0]]  # 입력층 추가

    for i in range(len(weights)):
        # 현재 레이어의 가중치와 이전 레이어의 활성화 값으로 다음 레이어 계산
        z = matrix_multiply(weights[i], activations[-1])  # 행렬 곱
        a = [[sigmoid(val[0])] for val in z]  # 활성화 함수 적용
        activations.append(a)  # 활성화 값 저장

    return activations

# 오차 계산 함수 (MSE)
def calculate_error(output, target):
    return sum([(output[i][0] - target[i]) ** 2 for i in range(len(target))]) / len(target)

# 메인 실행 코드
def main():
    # 초기 설정
    input_size = 64 * 64  # 입력층 크기 (64x64 이미지)
    base_path =  "C:/Users/asx12/OneDrive/바탕 화면/인공지능"  # 이미지 폴더 경로
    csv_file = "dataset_labels.csv"  # 라벨링 정보가 담긴 CSV 파일

    output_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/output_log.txt" #결과 출력 및 파일 저장
    weights_log_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/weights_log.txt" # 가중치 초기화 및 저장

    # 은닉층 개수 설정
    layer_num = int(input("은닉층 개수를 입력하세요 (3~14): "))
    if not (3 <= layer_num <= 14):
        raise ValueError("은닉층 개수는 3 이상, 14 이하이어야 합니다.")

    # 레이어별 노드 개수 설정
    node_counts = []
    print("\n각 레이어의 노드 개수를 입력하세요:")
    for i in range(layer_num + 2):  # 입력층, 은닉층, 출력층 포함
        count = int(input(f"Layer {i + 1} 노드 수: "))
        node_counts.append(count)

    # 데이터 로드
    print("\n데이터셋을 로드 중입니다...")
    inputs, targets = load_dataset(base_path, csv_file, input_size)

    # 난수 생성 시드 설정
    random_seed = 42  # 원하는 정수 값
    random.seed(random_seed)

    # 가중치 초기화
    weights = []
    for i in range(layer_num + 1):  # 레이어 간 연결 개수
        weights.append(initialize_matrix(node_counts[i + 1], node_counts[i]))
    
    # 시간 측정 시작
    start_time = time.time()
    
     # 파일로 출력 기록
    with open(output_log_file, mode="w", encoding="utf-8") as log_file:
        log_file.write("배치 학습 결과 로그\n")
        log_file.write("=====================\n")

        # 학습 및 결과 출력
        for idx, (input_data, target) in enumerate(zip(inputs, targets)):
            activations = forward_propagation([input_data], weights)
            output = activations[-1]
            error = calculate_error(output, target)

            # 터미널 출력
            #print(f"\n--- 샘플 {idx + 1} ---")
            #for i, activation in enumerate(activations):
            #    print(f"Layer {i + 1}: {[round(node[0], 4) for node in activation]}")
            #print(f"출력값: {[round(node[0], 4) for node in output]}")
            #print(f"목표값: {target}")
            #print(f"오차: {round(error, 4)}")

            # 로그 파일로 저장
            log_file.write(f"\n--- 샘플 {idx + 1} ---\n")
            for i, activation in enumerate(activations):
                log_file.write(f"Layer {i + 1}: {[round(node[0], 4) for node in activation]}\n")
            log_file.write(f"출력값: {[round(node[0], 4) for node in output]}\n")
            log_file.write(f"목표값: {target}\n")
            log_file.write(f"오차: {round(error, 4)}\n")
            
    # 시간 측정 종료
    elapsed_time = time.time() - start_time
    print(f"\n결과가 {output_log_file} 파일에 저장되었습니다.")
    print(f"총 실행 시간: {elapsed_time:.2f}초")
   
    # 가중치를 파일에 저장
    with open(weights_log_file, mode="w", encoding="utf-8") as weight_file:
        weight_file.write("초기화된 가중치 값 로그\n")
        weight_file.write("=====================\n")
        for layer_index, layer_weights in enumerate(weights):
            weight_file.write(f"\n--- Layer {layer_index + 1} 가중치 ---\n")
            for weight_row in layer_weights:
                weight_file.write(f"{[round(w, 4) for w in weight_row]}\n")

    print(f"초기화된 가중치가 {weights_log_file} 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
