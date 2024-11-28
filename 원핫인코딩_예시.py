import csv
import os

# 중복되지 않는 한글 글자 리스트 (155개의 글자로 확장 필요)
unique_chars = ['충', '청', '남', '도', '홍', '성', '군', '읍', '기', '길', '숙', '사', '호', '동', '박', '소', '윤', '김', '은', '옥', '영', '수', '종', '헌', '조', '인', '의', '경', '봉', '화', '천', '광', '역', '시', '연', '구', '아', '카', '데', '미', '로', '춘', '이', '는', '듣', '만', '하', '여', '가', '슴', '설', '레', '말', '다', '너', '두', '손', '을', '에', '대', '고', '물', '방', '같', '심', '장', '들', '어', '보', '라', '피', '끓', '뛰', '노', '거', '선', '관', '힘', '있', '것', '류', '를', '꾸', '며', '내', '려', '온', '력', '바', '투', '명', '되', '얼', '음', '과', '으', '지', '혜', '날', '우', '나', '갑', '속', '든', '칼', '니', '더', '면', '간', '마', '쓸', '랴', '싸', '죽', '뿐']  # 예시, 총 155개 글자로 확장 가능
num_chars = len(unique_chars)

# 원핫 인코딩 라벨 생성 함수
def get_one_hot_vector(index, size):
    vector = [0] * size
    vector[index] = 1
    return vector

# CSV 파일에 이미지 경로와 원핫 인코딩 라벨 저장
csv_filename = "dataset_labels_jimin.csv"
base_folder = "글자이미지생성_지민"

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    # 헤더 작성 (이미지 경로 + 각 글자에 대한 컬럼)
    header = ["image_path"] + unique_chars
    writer.writerow(header)
    
    # 각 글자 폴더 내 이미지 파일 경로와 원핫 인코딩 라벨 작성
    for i, char in enumerate(unique_chars):
        one_hot_vector = get_one_hot_vector(i, num_chars)
        char_folder = os.path.join(base_folder, char)
        
        # 해당 글자 폴더 내 모든 png 파일 탐색
        if os.path.exists(char_folder):
            for img_file in os.listdir(char_folder):
                if img_file.endswith(".bmp"):
                    image_path = os.path.join(char_folder, img_file)
                    writer.writerow([image_path] + one_hot_vector)
        else:
            print(f"{char_folder} 폴더가 존재하지 않습니다.")

print(f"{csv_filename} 파일이 생성되었습니다.")
