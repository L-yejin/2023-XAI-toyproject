import easyocr
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

import json
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import PyPDF2
import requests
from io import BytesIO

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'NanumGothic'
from matplotlib import rc
rc('font', family="AppleGothic")
font_path = './The_Jamsil_TTF/The Jamsil 2 Light.ttf'  # 여기에 한글 폰트 경로 지정

def default(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def should_merge(box1, box2, y_threshold):
    min_y_box1 = min([point[1] for point in box1])
    max_y_box2 = max([point[1] for point in box2])
    return abs(min_y_box1 - max_y_box2) <= y_threshold # bbox1의 min(y)와 bbox2의 max(y) 거리 비교

main_size = 'medium' # big, medium, small
folder_path = f"./DATA/{main_size}"  # 추출하려는 폴더 경로
# 폴더 내의 파일 목록 얻기
file_list = os.listdir(folder_path)
# 확장자가 "jpeg"인 파일 목록 출력
jpeg_files = [f'{folder_path}/{file}' for file in file_list if file.lower().endswith(".jpeg")]
# 파일 목록 출력
print('Num files: ',len(jpeg_files))
# print(jpeg_files)
########################################################################################################################################
# image_path = jpeg_files[0]

start_time = time.time()
reader = easyocr.Reader(['en', 'ko']) # OCR 엔진 생성

for image_path in jpeg_files:
    # 이미지 파일에서 텍스트 추출
    result = reader.readtext(image_path, blocklist=[':',"'",'_',';','"',' ','1','!',"",'{','}','*'], text_threshold=0.7, width_ths=0.8, add_margin=0.1)#, width_ths=0.37, add_margin=0.01)#, decoder='beamsearch', beamWidth=1)#, contrast_ths=0.5)
    ###################################################################################################
    # 병합 기준 설정
    y_threshold = 20  # 높이(y 좌표) 차이 허용 오차

    # 병합된 바운딩 박스와 텍스트 생성
    merged_boxes, merged_texts, merged_confidences = [], [], []
    result.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

    for box1, text1, confidence in result:
        if confidence >= 0.0:
            # print(text1, confidence)
            merged = False  # 병합 여부 확인
            for i, box2 in enumerate(merged_boxes):
                if should_merge(box1, box2, y_threshold):
                    merged_texts[i] += " " + text1

                    min_x = min(min([p[0] for p in box1]), min([p[0] for p in box2]))
                    min_y = min(min([p[1] for p in box1]), min([p[1] for p in box2]))
                    max_x = max(max([p[0] for p in box1]), max([p[0] for p in box2]))
                    max_y = max(max([p[1] for p in box1]), max([p[1] for p in box2]))

                    merged_boxes[i] = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
                    merged = True
                    break
            if not merged:
                merged_boxes.append(box1)
                merged_texts.append(text1)
        else:
            pass

    results, boxes, texts = [], [], []
    for box, text in zip(merged_boxes, merged_texts): # 결과를 리스트에 저장
        boxes.append(box)
        texts.append(text)

    for i in range(len(boxes)): # 딕셔너리를 리스트에 추가
        results.append([boxes[i], texts[i]])
    ########################################################################################################################################
    image = cv2.imread(image_path) # 이미지 로드
    image_pil = Image.fromarray(image)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Pillow용 이미지로 변환

    # 이미지에 텍스트와 경계 상자 시각화
    font = ImageFont.truetype(font_path, 20)
    draw = ImageDraw.Draw(image_pil)

    for i, text_info in enumerate(results):
        coordinates, text = text_info
        
        pts = [list(coord) for coord in coordinates] # 경계 상자 좌표를 구조체로 변환
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        draw.polygon(pts.flatten().tolist(), outline="blue") # 경계 상자 그리기
        x, y = int(coordinates[0][0]), int(coordinates[0][1]) # coordinates[0]
        draw.text((x, y - 30), text, font=font, fill=(0, 0, 255)) # 텍스트 추가
    ########################################################################################################################################
    save_json = []
    # 전체 이미지의 너비와 높이
    # print(image.shape[0],image.shape[1]) # h,w
    image_height, image_width = image.shape[0], image.shape[1]
    for bbox,text in results:
        dic = {}
        # Bbox의 너비와 높이 계산
        width = bbox[1][0] - bbox[0][0]
        height = bbox[2][1] - bbox[1][1]

        # 너비와 높이 비율 계산
        width_ratio = width / image_width
        height_ratio = height / image_height

        # 비율 기준 설정
        size_criteria = [
            {'name': 'small', 'width': 0.2, 'height': 0.2},
            {'name': 'medium', 'width': 0.4, 'height': 0.4},
            {'name': 'big', 'width': 1, 'height': 1}
        ]

        size = None
        for i, criteria in enumerate(size_criteria):
            if width_ratio <= criteria['width'] and height_ratio <= criteria['height']:
                size = criteria['name']
                break

        # 치우침 여부 기준 설정
        left_right_thresholds = {'left': 0.3, 'right': 0.7}
        top_bottom_thresholds = {'top': 0.3, 'bottom': 0.7}
        mid_x = (bbox[0][0] + bbox[1][0]) / 2
        mid_y = (bbox[0][1] + bbox[3][1]) / 2

        position = {'horizontal': None, 'vertical': None}
        
        if mid_x < image_width * left_right_thresholds['left']:
            position['horizontal'] = 'left'
        elif mid_x > image_width * left_right_thresholds['right']:
            position['horizontal'] = 'right'
        else:
            position['horizontal'] = 'middle'
        
        if mid_y < image_height * top_bottom_thresholds['top']:
            position['vertical'] = 'top'
        elif mid_y > image_height * top_bottom_thresholds['bottom']:
            position['vertical'] = 'bottom'
        else:
            position['vertical'] = 'middle'

        # print(f"텍스트: {text}")
        # print(f"비율: {size}")
        # print(f"가로 치우침 여부: {position['horizontal']}")
        # print(f"세로 치우침 여부: {position['vertical']}")
        # print()

        dic['text'] = text
        dic['bbox'] = bbox
        dic['vertical'] = position['vertical']
        dic['horizontal'] = position['horizontal']
        dic['size'] = size
        save_json.append(dic)

    ########################################################################################################################################
    # JSON 파일에 저장
    result_json = f'./ocr-json/{main_size}/{image_path.split("/")[-1].split(".jpeg")[0]}.json'
    with open(result_json, "w") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=1, default=default)

    # 결과 이미지 저장
    result_img = f'./ocr-img/{main_size}/{image_path.split("/")[-1]}'
    image_result = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_img, image_result)

###################################################################################################
end_time = time.time()
elapsed_time = end_time - start_time
print(f"실행에 걸린 시간: {elapsed_time} 초")