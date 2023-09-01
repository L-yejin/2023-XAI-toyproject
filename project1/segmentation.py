import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import segmentation_models_pytorch as smp
import ssl
import torch.nn as nn
import torchvision.transforms as transforms
import json
from collections import Counter
from PIL import Image

def predict_pt(pt_path, image_path):
    import torch
    import cv2
    device = torch.device('cpu')
    model = torch.load(pt_path, map_location=device)
    # (320, 480, 3)
    print('### image_path ### ',image_path)
    image = cv2.imread(image_path)
    print('### img size ### ', image.shape)
    # torch.Size([1, 3, 320, 480])
    # image_tensor = torch.from_numpy(image).to('cuda').permute(2,0,1).unsqueeze(0).float()
    image_tensor = torch.from_numpy(image).to('cpu').permute(2,0,1).unsqueeze(0).float()
    mask = model.predict(image_tensor)
    return mask
    
    
def find_least_frequent_indices(arr):
    # 배열에서 각 값들의 빈도수 계산
    unique_values, counts = np.unique(arr, return_counts=True)

    # 빈도수가 가장 적은 값 찾기
    least_frequent_value = unique_values[np.argmin(counts)]
    print('빈도수가 가장 적은 값: ',least_frequent_value)
    # 빈도수가 가장 적은 값과 동일한 모든 위치 찾기
    least_frequent_indices = np.argwhere(arr == least_frequent_value)

    return least_frequent_indices

def convert_to_1_if_above_threshold(arr, threshold=0.9):
    arr[arr >= threshold] = 1
    return arr


def get_all_jpg_paths(folder_path):
    jpg_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_path = os.path.join(root, file)
                jpg_paths.append(jpg_path)
    return jpg_paths


def threshold_tensor(tensor, threshold=0.5):
    # 조건에 따라 0과 1로 변환하는 함수
    def binary_conversion(x):
        return 1 if x >= threshold else 0

    # NumPy의 벡터화된 연산을 사용하여 텐서의 각 원소에 적용
    binary_tensor = np.vectorize(binary_conversion)(tensor)

    return binary_tensor



class ImageInference:
    def __init__(self):
        # 모델 초기화
        self.input_size = (320, 480)  # 모델에 맞게 입력 크기 지정 (예: 224x224)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])


    def load_model(self, class_name):
        self.class_name = class_name
        model_path = os.getcwd() # os.path.join(os.getcwd(),pt) #'.' # "/content/drive/MyDrive/X:AI_toy-proj/Final_pth"
        model_path = model_path+"/Final_pth/"+str(class_name.split("_")[-1])+".pth"
        # self.model = torch.load(model_path)
        self.model = torch.load(model_path, map_location=torch.device("cpu"))

        self.model.eval()

    def preprocess_image(self, image_path):
        # 이미지 전처리 (크기 조정, 정규화 등)
        image = Image.open(image_path)
        image = self.transform(image)
        #print(image.shape)
        image = image.unsqueeze(0)  # 배치 차원 추가 (1개의 이미지에 대해 추론하기 위해)
        return image

    def predict(self, image_path):
        # 이미지를 입력으로 받아 추론 결과를 반환하는 함수
        # preprocessed_image = self.preprocess_image(image_path).to("cuda")
        preprocessed_image = self.preprocess_image(image_path).to("cpu")
        self.test_image = preprocessed_image.squeeze(0).cpu().numpy()
        with torch.no_grad():
            prediction = self.model(preprocessed_image)
            prediction = prediction.squeeze(0).cpu()
            #print(prediction.shape)
        return threshold_tensor(prediction)

    def rle_encoding(self, mask):
        # RLE 인코딩 함수
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        #print(len(runs[1::2]))
        #print(len(runs[::2]))
        try:
            runs[1::2] -= runs[::2]
        except:
            runs[1::2] -= runs[:-1:2]
        result = ' '.join(str(x) for x in runs)
        result_dict[self.class_name] = result

        with open("result.json", 'w') as json_file:
            json.dump(result_dict, json_file)

    def visualization(self, mask):
        # 원본 이미지 로드
        # print(np.shape(mask))
        #original_image = Image.open(test_image_path)
        #print(np.shape(original_image))

        self.test_image = np.transpose(self.test_image, (1, 0, 2))

        # 마스크 시각화
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.test_image)
        plt.title('Original Image')
        plt.axis('off')


        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(mask,0), cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.show()



    def replace_background(self, index, img_path, background_path, mask, resize=False):
        # test_image_path = "/content/drive/MyDrive/X:AI_toy-proj/DATA/my_data/"
        #print(np.shape(self.test_image))
        #print(np.shape(mask))
        if index==0:
            self.test_image = np.transpose(self.test_image, (1, 2, 0))*255
        # self.test_image = cv2.imread(test_image_path + self.class_name +".jpg")
        self.test_image = cv2.imread(img_path)
        background_image = cv2.imread(background_path)
        mask_image = np.squeeze(mask,0)
        if resize != False:
            a= int(mask_image.shape[0]*resize)
            b= int(mask_image.shape[1]*resize)
            mask_image =  np.resize(mask_image, (a, b))
            #original_image = cv2.resize(original_image, (b,a))

        #print(mask_image)
        #print(self.test_image)
        # 지정한 값을 가지는 픽셀 위치 파악
        specified_value = 0  # 예시로 값이 255인 픽셀을 추출합니다
        #mask_image = convert_to_1_if_above_threshold(mask_image)
        positions = list((mask_image == specified_value).nonzero())

        positions[0] = [x for x in list(positions[0])]
        positions[1] = [x for x in list(positions[1])]
        #print(positions[0])
        #print(positions[1])

        pose=[[],[]]
        pose[0] = [y for y in list(positions[0])] # pose[0] = [y+loc[1] for y in list(positions[0])] 추출 객체 위치 조정 역할
        pose[1] = [x for x in list(positions[1])] # pose[1] = [x+loc[0] for x in list(positions[1])]

        #print(pose[0])
        #print(pose[1])
        # 추출한 부분에 해당하는 픽셀 값 넣어주기
        for y, x, y1, x1 in zip(pose[0], pose[1], positions[0], positions[1]):

            background_image[y, x] = self.test_image[y1, x1]
            # except:
            #     print(1)
            #     #print(y,x)

        # 추출된 부분 시각화 (optional)
        # cv2.imshow(background_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if index == False:
            new_image_name = f'OUTPUT/change_{self.class_name}.jpg'
        else:
            new_image_name = f'/content/drive/MyDrive/X:AI_toy-proj/output/image_{self.class_name}_{str(index)}.jpg'


        # 넘파이 배열을 이미지로 변환
        image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)  # BGR 형식으로 변환

        # 이미지 저장
        cv2.imwrite(new_image_name, background_image)
        print(f'이미지가 {new_image_name}으로 저장되었습니다.')