import os
import csv
from PIL import Image

# 기존 데이터셋 CSV 파일 경로와 새 데이터셋 CSV 파일 경로 설정
original_csv_file = "dataset_labels.csv"
new_csv_file = "dataset_labels_32x32.csv"

# 새 이미지가 저장될 기본 경로
resized_image_base_path = "./resized_images_32x32"
os.makedirs(resized_image_base_path, exist_ok=True)

# 새 CSV 파일에 리사이즈된 이미지 경로와 레이블 저장
with open(original_csv_file, newline="", encoding="utf-8") as csvfile, open(new_csv_file, "w", newline="", encoding="utf-8") as new_csvfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(new_csvfile)

    # 헤더를 복사
    header = next(reader)
    writer.writerow(header)

    # 각 이미지를 처리
    for row in reader:
        image_path = row[0]
        label = row[1:]

        # 기존 이미지 파일의 절대 경로 계산
        original_image_path = os.path.normpath(image_path)

        # 새 이미지 파일 경로 설정 (resized_image_base_path를 기준으로 새로운 파일 경로 설정)
        resized_image_path = os.path.join(resized_image_base_path, os.path.relpath(image_path, start=os.path.commonpath([original_image_path, resized_image_base_path])))

        # 이미지가 있는 폴더 생성
        os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)

        try:
            # 이미지 열기 및 리사이즈
            with Image.open(original_image_path) as img:
                resized_img = img.resize((32, 32), Image.LANCZOS)

                # 리사이즈된 이미지 저장
                resized_img.save(resized_image_path, format="BMP")

            # 새 CSV 파일에 리사이즈된 이미지 경로와 레이블 작성
            writer.writerow([resized_image_path] + label)
            print(f"{resized_image_path} 처리 완료")

        except FileNotFoundError as e:
            print(f"{original_image_path} 처리 중 오류 발생: {e}")
