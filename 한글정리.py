import re

text = """
충청남도 홍성군 홍성읍 청기 길 기숙사 호 동
박소윤
김은옥 박영수 박종헌 조인옥 김의경 인봉화
인천광역시 연수구 아카데미로
청춘 이는 듣기만 하여도 가슴이 설레는 말이다 청춘 너의 두 손을 가슴에 대고 
물방아 같은 심장의 고동을 들어보라 청춘의 피는 끓는다 끓는 피에 뛰노는 심장은 
거선의 기관같이 힘있다 이것이다 인류의 역사를 꾸며 내려온 동력은 바로 이것이다 
이성은 투명하되 얼음과 같으며 지혜는 날카로우나 갑 속에 든 칼이다 
청춘의 끓는 피가 아니더면 인간이 얼마나 쓸쓸하랴 얼음에 싸인 만물은 죽음이 있을 뿐이다
"""

# 한글만 추출하여 중복되지 않는 글자로 리스트 생성
korean_chars = re.findall(r'[가-힣]', text)

# 중복 제거하며 순서를 유지하는 코드
seen_chars = set()
unique_chars = []
for char in korean_chars:
    if char not in seen_chars:
        unique_chars.append(char)
        seen_chars.add(char)

# 결과 출력
print("중복되지 않는 한글 글자 리스트:", unique_chars)
print("중복되지 않는 한글 글자 수:", len(unique_chars))
