import numpy as np
import cv2

import pytesseract
import easyocr


from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence

import argostranslate.package
import argostranslate.translate

def pil_to_cv2(img):
  return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_gray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## my first basic attempt, works sometimes for big area
def get_boxes_aa(img):
  ret,th1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  dilation = cv2.dilate(th1,kernel,iterations = 4)
  cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  return cnts


def show_boxes(img, cnts):
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
  cv2.imshow("boxes", img)
  cv2.waitKey(0)


def get_rois(img, cnts):
  all_roi = []
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = img[y:y+h, x:x+w]
    all_roi.append(ROI)
  return all_roi

def ocr_tesseract(roi_list):
  #all_data = []
  all_strings = []
  for roi in roi_list:
    data = pytesseract.image_to_string(roi, lang='jpn', config='--psm 6')
    all_strings.append([''.join(s.split()) for s in data.split('\n') if s])
  return all_strings

def ocr_easyocr(roi_list):
  reader = easyocr.Reader(['ja','en'], gpu=False, verbose=False)
  #all_data = []
  all_strings = []
  for roi in roi_list:
    data = reader.readtext(roi)
    all_strings.append([a[1] for a in data])
  #return all_data
  return all_strings
 
def translate_marian(texts):
  model_name = 'Helsinki-NLP/opus-mt-ja-en'
  model = MarianMTModel.from_pretrained(model_name)
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  out_list = []
  for l in texts:
    tokens = tokenizer(list(l), return_tensors="pt", padding=True)
    translate_tokens = model.generate(**tokens, max_new_tokens=512)
    out_list.append([tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens])
  return out_list

def translate_argos(texts):
  argostranslate.package.install_from_path('./translate-ja_en-1_1.argosmodel')
  out_list = []
  for l in texts:
    out_list.append([argostranslate.translate.translate(tt, "ja", "en") for tt in l])
  return out_list
