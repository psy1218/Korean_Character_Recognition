# 시그모이드 함수 정의
def sigmoid(x):
    return [1 / (1 + (2.71828 ** -xi)) for xi in x]

# 행렬 초기화 함수
def initialize_matrix(rows, cols, value=0):
    return [[value] * cols for _ in range(rows)]

# 행렬 곱 함수
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("행렬 A의 열 수와 B의 행 수가 일치해야 합니다.")

    result = initialize_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# 각 레이어의 노드 값을 입력받는 함수
def input_nodes(layer_num):
    node_num = []
    nodes = []

    for i in range(layer_num):
        num = int(input(f"Layer {i+1}의 node 개수를 입력하세요: "))
        node_num.append(num)

        if i == 0:
            # 첫 번째 레이어의 노드 값 입력
            layer_values = []
            for j in range(num):
                value = float(input(f"Layer {i+1}의 노드 {j+1}의 값을 입력하세요: "))
                layer_values.append(value)
            nodes.append([[v] for v in layer_values])  # 1차원 리스트를 열 벡터로 변환
        else:
            # 첫 번째 레이어가 아니면 빈 배열 할당
            nodes.append(initialize_matrix(num, 1))

    return node_num, nodes

# 각 레이어 간의 가중치 입력 함수
def input_weights(layer_num, node_num):
    weights = []

    for i in range(layer_num - 1):
        from_nodes = node_num[i]  # 현재 레이어의 노드 수
        to_nodes = node_num[i + 1]  # 다음 레이어의 노드 수

        weight_matrix = initialize_matrix(from_nodes, to_nodes)

        print(f"\nLayer {i+1} -> Layer {i+2} 가중치 입력:")
        for j in range(from_nodes):
            for k in range(to_nodes):
                weight_matrix[j][k] = float(input(f"가중치[{j+1}][{k+1}]: "))

        weights.append(weight_matrix)

    return weights

# 다음 레이어의 노드 값을 계산하는 함수
def next_nodes(prev_values, weights):
    # 행렬 곱으로 계산
    return matrix_multiply(weights, prev_values)

# 메인 함수
def main():
    layer_num = int(input("레이어 개수를 입력하세요: "))

    # 각 레이어의 노드 값 입력
    node_num, nodes = input_nodes(layer_num)

    # 가중치 입력
    weights = input_weights(layer_num, node_num)

    # 순전파 수행
    for i in range(layer_num - 1):
        nodes[i + 1] = next_nodes(nodes[i], weights[i])

        # 시그모이드 함수 적용 (활성화 함수)
        nodes[i + 1] = [[sigmoid([val[0]])[0]] for val in nodes[i + 1]]

    # 결과 출력 (각 레이어의 노드 값 출력)
    for i in range(layer_num):
        print(f"\nLayer {i+1}의 노드 값: {[node[0] for node in nodes[i]]}")

if __name__ == "__main__":
    main()
