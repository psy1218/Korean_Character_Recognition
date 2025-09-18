import os
import re
import matplotlib.pyplot as plt

# 각 에폭 폴더를 순회하며 mse와 정확도 값을 추출합니다.
base_path = "C:/Users/asx12/OneDrive/바탕 화면/인공지능/all_3fonts_0.0001"
epochs = []
mse_values = []
train_accuracies = []
test_accuracies = []

for epoch_num in range(1, 45):
    epoch_folder = os.path.join(base_path, f"epoch_{epoch_num}")
    details_file = os.path.join(epoch_folder, f"epoch_{epoch_num}_details.txt")
    results_file = os.path.join(epoch_folder, f"epoch_{epoch_num}_results.txt")

    if not os.path.exists(details_file):
        print(f"경고: {details_file} 파일이 존재하지 않습니다.")
        continue

    if not os.path.exists(results_file):
        print(f"경고: {results_file} 파일이 존재하지 않습니다.")
        continue

    # details.txt에서 MSE 값 추출
    with open(details_file, 'r', encoding='utf-8') as f:
        details_content = f.read()
        mse_match = re.search(r"--- MSE \(Mean Squared Error\) ---\s*([\d.]+)", details_content)
        if mse_match:
            mse = float(mse_match.group(1))
            mse_values.append(mse)
        else:
            print(f"경고: {details_file}에서 MSE 값을 찾을 수 없습니다.")

    # results.txt에서 훈련 정확도와 테스트 정확도 추출
    with open(results_file, 'r', encoding='utf-8') as f:
        results_content = f.read()
        train_match = re.search(r"훈련 정확도:\s*([\d.]+)%", results_content)
        test_match = re.search(r"테스트 정확도:\s*([\d.]+)%", results_content)
        if train_match and test_match:
            train_accuracy = float(train_match.group(1))
            test_accuracy = float(test_match.group(1))
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        else:
            print(f"경고: {results_file}에서 정확도 값을 찾을 수 없습니다.")

    epochs.append(epoch_num)

# 그래프 그리기
if len(epochs) == len(mse_values) == len(train_accuracies) == len(test_accuracies):
    plt.figure(figsize=(14, 8))

    # 사용자로부터 제목 입력받기
    user_title = input("그래프의 제목을 입력하세요: ")
    plt.suptitle(user_title, fontsize=16)  # 제목 추가

    # MSE 변화 그래프
    plt.subplot(3, 1, 1)
    plt.plot(epochs, mse_values, marker='o', color='b', label='MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE over Epochs')
    plt.legend()

    # 훈련 정확도 변화 그래프
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_accuracies, marker='o', color='g', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.legend()

    # 테스트 정확도 변화 그래프
    plt.subplot(3, 1, 3)
    plt.plot(epochs, test_accuracies, marker='o', color='r', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목을 위한 공간 확보
    plt.show()
else:
    print("에폭, MSE, 훈련 정확도, 테스트 정확도의 데이터 개수가 일치하지 않습니다.")
