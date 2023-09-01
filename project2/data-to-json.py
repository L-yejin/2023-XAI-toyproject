import os
import json

main_size = 'medium' # big, medium, small
folder_path = f"./DATA/{main_size}"  # 추출하려는 폴더 경로
# 폴더 내의 파일 목록 얻기
file_list = os.listdir(folder_path)
# 확장자가 "jpeg"인 파일 목록 출력
jpeg_files = [f'{folder_path}/{file}' for file in file_list if file.lower().endswith(".jpeg")]
# 파일 목록 출력
print('Num files: ',len(jpeg_files))

def extract_number(filename):
    base_name = filename.split('/')[-1]  # '/'로 분리 후 마지막 요소 선택
    number_str = base_name.split('_')[0]  # '_'로 분리 후 첫번째 요소 선택
    return int(number_str)  # 숫자 문자열을 정수로 변환

# 위에서 정의한 함수를 이용해 리스트를 정렬
jpeg_files.sort(key=extract_number)

# print(jpeg_files)

dic = {}
for name in jpeg_files:
    file_name = name.split('/')[-1]

    first_txt, second_txt, third_txt = '','',''
    first_txt += file_name.split('_')[1][0]
    second_txt += file_name.split('_')[1][1]
    third_txt += file_name.split('_')[1][2]

    first_txt += file_name.split('_')[2][0]
    second_txt += file_name.split('_')[2][1]
    third_txt += file_name.split('_')[2][2]

    first_txt += file_name.split('_')[3][0]
    second_txt += file_name.split('_')[3][1]
    third_txt += file_name.split('_')[3][2]

    # vertical('top','middle','bottom'), horizontal('left','middle','right'), size('small','medium','big')
    # print(first_txt, second_txt, third_txt)

    if second_txt == '000':
        cycle = [first_txt]
    elif third_txt == '000':
        cycle = [first_txt, second_txt]
    elif third_txt != '000':
        cycle = [first_txt, second_txt, third_txt]
    
    num=0
    middle_dic = {}
    for txt in cycle:
        final_dic = {}
        num += 1

        if txt[0] == '1':
            final_dic['vertical'] = 'top'
        elif txt[0] == '2':
            final_dic['vertical'] = 'middle'
        elif txt[0] == '3':
            final_dic['vertical'] = 'bottom'

        if txt[1] == '1':
            final_dic['horizontal'] = 'left'
        elif txt[1] == '2':
            final_dic['horizontal'] = 'middle'
        elif txt[1] == '3':
            final_dic['horizontal'] = 'right'

        if txt[2] == '1':
            final_dic['size'] = 'small'
        elif txt[2] == '2':
            final_dic['size'] = 'medium'
        elif txt[2] == '3':
            final_dic['size'] = 'big'
        
        middle_dic[f'txt{num}'] = final_dic

    # if file_name == '37_330_310_320.jpeg':
    #     print(first_txt)
    #     print(second_txt)
    #     print(third_txt)
    #     print(middle_dic)

    # print(middle_dic)
    
    # middle_dic['vertical']#'top','bottom','middle'
    # middle_dic['horizontal']
    # middle_dic['size']

    dic[name] = middle_dic

# print(dic)
# JSON 파일에 저장
result_json = f'./DATA/{main_size}_data.json'
with open(result_json, "w") as f:
    json.dump(dic, f, ensure_ascii=False, indent=1)