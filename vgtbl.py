#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageDraw
import json
from base64 import b64decode, b64encode
from io import BytesIO
import pytesseract
import cv2
import numpy as np
from pytesseract import Output
import argostranslate.package
import argostranslate.translate
from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence


class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]


argostranslate.package.install_from_path('./translate-ja_en-1_1.argosmodel')
marian_ja_en = Translator('ja', 'en')

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()


    def do_GET(self):
        self._set_headers()


    def do_HEAD(self):
        self._set_headers()


    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print("\n\n--------------------\n\n")
        print(self.headers)
        print("\n\n--------------------\n\n")
        bbb = json.loads(post_data.decode())['image']
        image = BytesIO(b64decode(bbb))
        im = Image.open(image)
        cv2_img = np.array(im)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        dilation = cv2.dilate(th1,kernel,iterations = 4)
        cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        ROI_number = 0
        for c in cnts:
          area = cv2.contourArea(c)
          if area > 10000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = img[y:y+h, x:x+w]
            cv2.imshow(f'ROI_{ROI_number}',ROI)
            ROI_number += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        data = pytesseract.image_to_string(ROI, lang='jpn', config='--psm 6').lower()
        oo = []
        print('------------')
        for ss in [s for s in data.split('\n') if s]:
          rr = ''.join(ss.split())
          oo.append(rr)
          tt = argostranslate.translate.translate(rr, "ja", "en")
          uu = marian_ja_en.translate([rr])
          print(f"IN: {rr}")
          print(f"ARGOS: {tt}")
          print(f"MARIAN: {uu}")
          print('------------')
        
        ttt = argostranslate.translate.translate(" ".join(oo), "ja", "en")
        uuu = marian_ja_en.translate([" ".join(oo)])
        print(f"WHOLE-OCR {oo}")
        print('------------')
        print(f"WHOLE-ARGOS {ttt}")
        print('------------')
        print(f"WHOLE-MARIAN {uuu}")
        print('------------')


def run(server_class=HTTPServer, handler_class=S, port=4404):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
