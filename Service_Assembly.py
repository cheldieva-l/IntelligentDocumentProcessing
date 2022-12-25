#dashboard.heroku.com/new-app
#streamlit run Service_Assembly.py
#https://bgremoval.streamlit.app/
# cd C:\Users\user\Desktop\Education\IntelligentDocumentProcessing
#https://www.google.com/search?q=streamlit&biw=1904&bih=952&tbm=vid&sxsrf=ALiCzsYjpEw1IYs9MM-txlwxeisg4NkI7Q%3A1671872547210&ei=I8CmY7XBDMKQrgTU7Z2QDA&ved=0ahUKEwj13qTK8pH8AhVCiIsKHdR2B8IQ4dUDCA0&uact=5&oq=streamlit&gs_lcp=Cg1nd3Mtd2l6LXZpZGVvEAMyBAgAEEMyBAgAEEMyBAgAEEMyBAgAEEMyBAgAEEMyBAgAEEMyBAgAEEMyBQgAEIAEMgUIABCABDIECAAQQ1AAWABgmgFoAHAAeACAAUmIAUmSAQExmAEAoAEBwAEB&sclient=gws-wiz-video#fpstate=ive&vld=cid:3392f1d3,vid:JwSS70SZdyM

import streamlit as st

st.write("""
 Service_Assembly App
 """)

#1 Гугл диск
#from google.colab import drive
#drive.mount('/content/drive')

repo_folder = 'C:\\Users\\user\\Desktop\\Education\\'

reqs_path = repo_folder + 'IntelligentDocumentProcessing\\requirements.txt'
#!pip3 install -r {reqs_path}

import math

import dill
import numpy as np
import sys
model_folder = repo_folder + 'IntelligentDocumentProcessing\\'
base_folder = repo_folder + 'IntelligentDocumentProcessing\\Resources\\e_Service_Deployment\\'  # import utils /
sys.path.append(base_folder)
sys.path.append(repo_folder + 'IntelligentDocumentProcessing\\Resources\\')  # from a_Text_Detection.utils import
sys.path.append(repo_folder)  # from IntelligentDocumentProcessing.Resources.a_Text_Detection.utils import

import torch
from typing import Union, List

import albumentations as A
from albumentations import BasicTransform, Compose, OneOf
from albumentations.pytorch import ToTensorV2

import torch.nn as nn

from a_Text_Detection.utils import Postprocessor, DrawMore


import torch
from typing import List, Any, Tuple
from itertools import groupby

import cv2
import matplotlib.pyplot as plt

import math

import dill
from sklearn.cluster import DBSCAN

from c_Layout_Analisys.utils import sort_boxes_top2down_wrt_left2right_order, sort_boxes, fit_bbox, Paragraph

from c_Layout_Analisys.utils import resize_aspect_ratio

from utils import prepare_crops

from ner_model import sentence_split

from ner_model import inference_ner_model
from ipymarkup import show_span_line_markup

# КОД ДЛЯ СТУДЕНТА

def merge_predictions(document_predictions: List[Tuple[str, list]]) -> Tuple[str, list]:
    markup = []
    text = ""
    # 
    # Дополнение по коду
    # 
    curr_pos = 0
    for sentence, tags in document_predictions:
        markup.extend([(s + curr_pos, e + curr_pos, t) for s, e, t in tags])
        text += sentence + ' '
        curr_pos = len(text)
    text = text.strip() 
    return text, markup

image_fpath = model_folder+'/821284f7-4c42-491e-b85d-9d37a2ce7a56.jpeg'

image = cv2.imread(image_fpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


device = 'cpu'#'cuda:0'
max_image_size = 2048

image_fpath = model_folder+'821284f7-4c42-491e-b85d-9d37a2ce7a56.jpeg'
#'./team_idp/ocr_service/ner_sample/821284f7-4c42-491e-b85d-9d37a2ce7a56.jpeg'

image = cv2.imread(image_fpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized, _, _ = resize_aspect_ratio(image, square_size=max_image_size, interpolation=cv2.INTER_LINEAR)

# КОД ДЛЯ СТУДЕНТА
# сюда необходимо вставить код токенайзера из тетрадки по распознаванию текста
class TokenizerForCTC:
    #pass
    def __init__(self, tokens: List[Any]):  
        """
        Класс для преобразования текста в тензор для подачи в CTCLoss и преобразования из тензора в текст.

        Args:
            tokens: список токенов (символов) из алфавита
        """  
        self.char_idx = {}
        self.dict={}
        self.num_val={}        
        self.tokens=tokens

        for i, char in enumerate(tokens):
          self.char_idx[char] = i + 1
        for i in range(len(self.tokens)):
          self.num_val[i+1] = self.tokens[i]  

    def encode(self, text: str) -> Tuple[torch.Tensor, int]:
        """
        Метод для преобразования текста в тензор для CTCLoss.

        Args:
            text: текст

        Returns:
            токенизированный текст в виде тензора, длина входного текста
        """
        returns=[]
        for i in text:
           returns.append([self.char_idx[i]])

        res=[torch.LongTensor(returns)]
        res.append(len(returns))

        return res

    def decode(self, preds: List[Any]) -> str:
        """
        Метод для перевода токенизированного текста в обычный текст.

        Args:
            preds: предсказания модели с одной строкой текста

        Returns:
            уверенность модели в виде вероятностей; строка с распознанным текстом
        """
        returns = ''.join([self.num_val[i] for i in [k[0] for k in groupby(preds) if k[0]!=0]])
                           
        return returns
		
	# КОД ДЛЯ СТУДЕНТА
def line_recognition_pipeline(
    image: np.ndarray,
    device: str,
    #**kwargs
    line_model: nn.Module,
    line_transform: Union[BasicTransform, Compose, OneOf],
    line_postprocessor: Postprocessor,
    paragraph_model: Any,
    ocr_model: nn.Module,
    ocr_transform: Union[BasicTransform, Compose, OneOf],
    ocr_tokenizer: TokenizerForCTC,
    ocr_batch_size: int = 1,
    ocr_target_height: int = 32,
    ocr_pad_value: int = 0    
) -> List[Paragraph]:
    #pass
    lines = line_detector_inference(line_model, image, line_transform, line_postprocessor, device)
    pred_bboxes = [line.bbox for line in lines]
    line_crops = prepare_crops(image, pred_bboxes)

    line_labels = ocr_inference(ocr_model, line_crops, ocr_transform, ocr_tokenizer, device, ocr_batch_size, ocr_target_height, ocr_pad_value)
    for line, line_label in zip(lines, line_labels):
      line.label = line_label

    paragraphs = paragraph_model.find_paragraphs(lines)
    for para in paragraphs:
      para.label = ' '.join([line.label.strip() for line in para.items])

    return paragraphs  	
		
from c_Layout_Analisys.utils import Line

# необходимо вставить сюда инференс из тетрадки по layout
# КОД ДЛЯ СТУДЕНТА
def line_detector_inference(
    model: nn.Module, 
    image: np.ndarray, 
    transform: Union[BasicTransform, Compose, OneOf],
    postprocessor: Postprocessor,
    device: str = 'cpu',
) -> List[Line]:
    #pass
    # подготовка изображения (c помощью preprocessor)
    h, w, _ = image.shape
    image_changed = transform(image=image)['image']
    image_changed = torch.Tensor(image_changed[np.newaxis,:])
    # предсказание модели (с помощью model)
    with torch.no_grad():
        prediction = model(image_changed.to(device))
    # постпроцессинг предсказаний (с помощью postprocessor)
    pred_image = prediction[0].cpu().detach().numpy()
    bboxs = postprocessor(w, h, pred_image, return_polygon=False)[0][0]
    # нормализация bounding box'ов по высоте и ширине
    bboxs_arrays = np.array([np.array(x) for x in bboxs])
    normalized_bboxs=bboxs_arrays/np.array([w, h])
    # создание списка объектов типа Line 
    res = [Line(bbox=x, normalized_bbox=y) for x, y in zip(bboxs_arrays, normalized_bboxs)]

    return res

size = 512
transform =  A.Compose([
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.,
        p=1.,
    ),
    ToTensorV2(p=1.)])
postprocessor =  Postprocessor(
    unclip_ratio=1.5,
    binarization_threshold=0.3,
    confidence_threshold=0.7,
    min_area=1,
    max_number=1000)

import torch.nn.functional as F

from utils import batchings
from b_Optical_Character_Recognition.utils import resize_by_height

# КОД ДЛЯ СТУДЕНТА
def ocr_inference(
    model: nn.Module, 
    image: np.ndarray, 
    transform: Union[BasicTransform, Compose, OneOf],
    tokenizer: TokenizerForCTC, 
    device: str = 'cpu',
    batch_size: int = 1,
    target_height: int = 32,
    pad_value: int = 0
) -> List[str]:
    #pass
    strings = []
    # Разбить входящие изображения с помощью метода batchings
    for batch in batchings(image, batch_size):
        # Каждую картинку в батче преобразовать с помощью resize_by_height
        batch = [resize_by_height(img, target_height) for img in batch]
        # Вычислить максимальную ширину изображения в батче
        max_width = np.max([img.shape[1] for img in batch])
        # Добить все изображения в батче до одной ширины значениями pad_value
        batch = [cv2.copyMakeBorder(
            img, 
            0, 0, 
            0, 
            (max_width - img.shape[1] + pad_value),
            cv2.BORDER_CONSTANT
        ) 
        for img in batch]
        # подготовка изображения
        batch_changed = [
            transform(image=img)['image']
            for img in batch
        ]
        # Привести все изображения к тензорам и объединить в один тензор через torch.stack
        batch = torch.stack([torch.Tensor(img) for img in batch_changed])
        # предсказание модели (с помощью model)
        with torch.no_grad():
            preds = model(batch).to(device)
        # постпроцессинг предсказаний
        preds = preds.log_softmax(dim=2)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).detach().cpu().numpy()
        pred_labels = [tokenizer.decode(p) for p in preds]
        string_in_batch = [''.join(pl) for pl in pred_labels]
        strings += string_in_batch
    return strings
transform = A.Compose([
    A.Normalize(
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        max_pixel_value=255.,
        p=1.,
    ),
    ToTensorV2(p=1.)])

# КОД ДЛЯ СТУДЕНТА

class Pipeline:
    
    def __init__(self):
        """
        Здесь нужно проинициализировать все модели, с помощью которых мы будем
        извлекать информацию и распознавать документы.
        """
        #pass
        self.device = "cpu"

        line_model_path = model_folder+'la.jit'
        self.line_model = torch.jit.load(line_model_path, map_location=torch.device(self.device))
        self.line_model.eval();

        self.transform = A.Compose([
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255.,
                p=1.,
            ),
            ToTensorV2(p=1.)
        ])
        self.postprocessor = Postprocessor(
            unclip_ratio=1.5,
            binarization_threshold=0.3,
            confidence_threshold=0.7,
            min_area=1,
            max_number=1000
        )
        
        with open(model_folder+'pf.pkl', 'rb') as r:
            self.paragraph_finder = dill.load(r)
        
        ocr_model_fpath = model_folder+'ocr.jit'
        self.ocr_model = torch.jit.load(ocr_model_fpath, map_location=torch.device(self.device))
        self.ocr_model.eval();
        
        punct = " !\\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{|}~«»№"
        digit = "0123456789"
        cr = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё"
        latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alphabet = punct + digit + cr + latin

        self.tokenizer = TokenizerForCTC(list(alphabet))
        
        self.model = torch.jit.load(model_folder+'ner_rured.jit')         

    def predict(self, image) -> dict:
        """
        Изображение уже в памяти. Ресайзим, детектируем, распознаем, собираем в единый текст.
        Сегментируем на предложения, извлекаем, объединяем и форматирем в словарь.
        Return:  {
            "text" : "Успешные результаты распознавания текста"
            "entities: [("Сущность 1", "Успешные результаты"), ("Сущность 2", "распознавания текста")]
        }
        Поле сущности так можете дополнять (координаты в текста/на исхображении),
        если захотите визуализировать результаты.
        """
        #pass
        ## читаем изображение в память
        #image = cv2.imread(image_fpath)
        ## меняем каналы
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## ресайзим 
        image_resized, _, _ = resize_aspect_ratio(image, square_size=max_image_size, interpolation=cv2.INTER_LINEAR)#
        ## вызов пайплайна распознавания для каждого изображения
        lines = line_recognition_pipeline(
            image=image_resized, line_model=self.line_model, 
            line_transform=self.transform, line_postprocessor=self.postprocessor,
            paragraph_model=self.paragraph_finder, 
            ocr_model=self.ocr_model, ocr_transform=self.transform,
            ocr_tokenizer=self.tokenizer, ocr_batch_size=8,
            device=self.device
        )
        
        ## получение полнотекста для документа
        rec_text = ' '.join([line.label for para in lines for i, line in enumerate(para.items)])
        ## сегментация и форматирование примеров для подачи в модель извлечения 
        samples = sentence_split(rec_text)
        ## инференс NER модели
        predictions_by_sentence = inference_ner_model(samples, self.model, self.device, batch_size=4, num_workers=2)
        ## объединение выхода из модели извлечения
        united_text, predicted_entities = merge_predictions(predictions_by_sentence)

        return {
            "recognized_text" : united_text, 
            "entities": predicted_entities
        }  

st.write("""
  Проверка имплементации
 """)

## Проверка имплементации
pipe = Pipeline()

st.write(pipe)

model_result = pipe.predict(image)

st.write("""
  model_result
 """)
st.write(model_result)
 
assert all([True if i in {"recognized_text", "entities"} else False for i in model_result.keys()]), "Some keys not found in model result"


print(model_result)

st.write(model_result)





