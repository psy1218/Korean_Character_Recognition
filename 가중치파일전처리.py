import re

weights_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/all_training_results_plz_size/epoch_20/epoch_20_details.txt"
output_file = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/all_training_results_plz_size/processed_weights_with_labels.txt"

# 가중치 파일에서 '--- Updated Weights ---' 이후의 가중치 값들을 읽어와서 float 값으로 변환하여 저장하는 함수
def process_weights_file(weights_file, output_file):
    with open(weights_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_weights_start = False
    current_layer = None
    processed_weights = []

    for line in lines:
        line = line.strip()
        
        if '--- Updated Weights ---' in line:
            updated_weights_start = True
            continue

        if updated_weights_start:
            # 레이어 정보 추출
            layer_match = re.match(r'--- Layer (\d+) Weights ---', line)
            if layer_match:
                current_layer = int(layer_match.group(1))
                processed_weights.append(f'Layer {current_layer} Weights:')
                continue

            # 가중치 값이 있는 줄인 경우
            if line.startswith('[np.float64('):
                # [np.float64(...), ...] 형식을 float 값으로 변환
                weights = re.findall(r'np\.float64\(([-\d\.e]+)\)', line)
                float_weights = [float(w) for w in weights]
                processed_weights.append(' '.join(map(str, float_weights)))

    # 결과를 output 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_weights:
            f.write(line + '\n')

# 가중치 파일 처리
process_weights_file(weights_file, output_file)
