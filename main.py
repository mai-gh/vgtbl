from PIL import Image

import vgtbl as v



f = v.pil_to_cv2(Image.open('./in.png'))
fg = v.cv2_gray(f)
cc = v.get_boxes_aa(fg)
#v.show_boxes(f,cc)

rr = v.get_rois(fg, cc)

oo_t = v.ocr_tesseract(rr)
oo_e = v.ocr_easyocr(rr)

print('================================')
print(f'{oo_e=}')
print('================================')
print(f'{oo_t=}')
print('================================')

tt_em = v.translate_marian(oo_e)
tt_tm = v.translate_marian(oo_t)
print('================================')
print(f'{tt_em=}')
print('================================')
print(f'{tt_tm=}')
print('================================')

tt_ea = v.translate_argos(oo_e)
tt_ta = v.translate_argos(oo_t)
print('================================')
print(f'{tt_ea=}')
print('================================')
print(f'{tt_ta=}')
print('================================')


