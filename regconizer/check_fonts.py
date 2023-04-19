""" Created by MrBBS """
# 12/20/2021
# -*-encoding:utf-8-*-

from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import cv2
import numpy as np

list_font = Path('fonts').rglob('*.[ot][tt]*')
s = ' 3422 (Gửi người tôi yêu) `dự kiến` <quay> !vào# @cuối& %tháng* Ba. "từ? giờ~ -đến_ cuối tháng còn có mười ngày"'
for p in list_font:
    font = ImageFont.truetype(p.as_posix(), size=22, encoding='utf-8')
    wT, hT = font.getsize(s)
    txt = Image.new('L', (wT + 25, hT + 25))
    ImageDraw.Draw(txt).text((12, 12), s, 255, font)
    cv2.imshow(p.name, np.array(txt))
    cv2.waitKey()
    cv2.destroyAllWindows()
