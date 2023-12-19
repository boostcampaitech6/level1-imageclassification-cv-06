import torch

from pathlib import Path
from rembg import remove, new_session
import os
from PIL import Image
import shutil
from tqdm import tqdm

# import onnxruntime as ort
# providers = ort.get_available_providers()
# print("Available Providers:", providers)

session = new_session(providers=['CUDAExecutionProvider'])

# train data 새로 만들기
print('train start')

old_path = '/data/ephemeral/home/data/train/images/' # train 이미지들 경로
new_path = '/data/ephemeral/home/new_data/train/images/' # 새로 생성된 train 이미지를 넣을 경로

for folder in tqdm(os.listdir(old_path), desc='Processing', unit='Folder'):
    path_ = old_path + folder
    # print(path_)
    if '._' in path_:
        continue
    for file in os.listdir(old_path + folder):
        # print(file)

        input_path = old_path + folder + '/' + file
        output_path = new_path + folder + '/' +  file
        
        path_1, _ = os.path.splitext(input_path)
        # print(path_1, _)
        if '._' in path_1:
            continue
        os.makedirs(new_path + folder, exist_ok = True)
        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)

# csv파일 복제해서 new_data/train에 옮기기
source_ = '/data/ephemeral/home/data/train/train.csv'

destination_ = '/data/ephemeral/home/new_data/train/train.csv'

shutil.copyfile(source_, destination_)
print('train end')

# eval data 새로 만들기
print('eval start')
cnt = 0
# print(len(os.listdir('/data/ephemeral/home/data/eval/images/')))

for file in os.listdir('/data/ephemeral/home/data/eval/images/'):
    # print(file)
    cnt += 1
    input_path = '/data/ephemeral/home/data/eval/images/' + file # validation에 있는 이미지 파일 경로
    output_path = '/data/ephemeral/home/new_data/eval/images/' + file # 새로운 validation 이미지를 넣을 이미지 파일 경로
    path, _ = os.path.splitext(input_path) 
    if '._' in path:
        continue
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)

# csv파일 복제해서 new_data/eval에 옮기기
source = '/data/ephemeral/home/data/eval/info.csv'

destination = '/data/ephemeral/home/new_data/eval/info.csv'
 
shutil.copyfile(source, destination)
print('eval end')

