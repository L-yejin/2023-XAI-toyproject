# 아에 갯수가 안맞으면 False
# 갯수가 일치한 것 중 size / vertical / horizontal 3가지 고려해서 갯수만큼 다 매칭되면 True / 아니면 False
import json

correct_count = 0
wrong_count = 0

for main_size in ['big', 'medium', 'small']:
    with open(f'./DATA/{main_size}_data.json', 'r') as f:
        json_data = json.load(f)

    for i, key in enumerate(json_data.keys()):
        file_name = key.split('/')[-1].split('.jpeg')[0]
        with open(f'./ocr-json/{main_size}/{file_name}.json', 'r') as f:
            predict_json = json.load(f)
        if len(json_data[key]) != len(predict_json): # 갯수가 안맞는 경우
            wrong_count+=1
            # print(key)
            pass
        else:
            # print(json_data[key])
            # print(predict_json)
            match_found = False
            for kkey, value in json_data[key].items():
                for predict_value in predict_json:
                    if all(value[k] == predict_value[k] for k in ['size', 'vertical', 'horizontal']):
                        # print(value['size'],value['vertical'],value['horizontal'])
                        # print(predict_value['size'],predict_value['vertical'],predict_value['horizontal'])
                        match_found = True
                        break
            if match_found:
                correct_count+=1
            else:
                wrong_count+=1
                # print(key)
                # print(json_data[key])
                # print(predict_json)

print('wrong_count: ', wrong_count)
print('correct_count: ', correct_count)
percent = correct_count/(correct_count+wrong_count)
print(f'{percent*100}%')