""" Created by MrBBS """
# 4/14/2021
# -*-encoding:utf-8-*-

import os
import random
import string
import warnings
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from termcolor import colored
# from computer_text_generator import ComputerTextGenerator
from background_generator import BackgroundGenerator
from computer_text_generator import ComputerTextGenerator
from distorsion_generator import DistorsionGenerator
from matplotlib import pyplot as plt
import uuid
warnings.filterwarnings("ignore", category=UserWarning)  # Tắt UserWarning
os.system('color')
chars = open('char.txt', 'r', encoding='utf-8').read().split('\t')


def print_error(name, mess=None):
    print(colored('\n[ERROR]', 'red'), name + ':', mess)


def print_info(name, isStart=True):
    if isStart:
        print(colored('\n[INFO]', 'cyan'), f'{name}: Start')
    else:
        print(colored('\n[INFO]', 'cyan'), f'{name}: End')


def get_font(path):
    try:
        print_info('get_font')
        fonts = []
        for p in Path(path).rglob('*.[ot][tt]*'):
            font = ImageFont.truetype(p.as_posix(), size=35, encoding='utf-8')
            fonts.append(font)
        print_info('get_font', False)
        return fonts
    except Exception as e:
        print_error('get_font', e)


def random_color(light=False, dark=False, hex_code=True):
    """
    Tạo màu ngẫu nhiên
    :param dark:
    :param light: tạo màu sắc sáng
    :param hex_code: tạo mã màu hex
    :return: (light = False - hex_code = True) mã màu HEX của màu sắc sáng / (light = True) mã màu RGB
    """
    if dark:
        gamma = random.randint(0, 68)
        return gamma, gamma, gamma
    if light:
        return random.randint(128, 256), random.randint(128, 256), random.randint(128, 256)
    else:
        if hex_code:
            return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        else:
            return random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)


def get_gradient_2d(start, stop, width, height, is_horizontal):
    """
    Tạo ảnh gradient 2D với thang màu trắng đen
    :param start: Màu sắc bắt đầu
    :param stop: Màu sắc kết thúc
    :param width: Chiều rộng
    :param height: Chiều cao
    :param is_horizontal: Bool - True: Gradient ngang - False: Gradient dọc
    :return: Ảnh gradient đen trắng
    """
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    """
    Tạo ảnh gradient 3D
    :param width: Chiều rộng
    :param height: Chiều cao
    :param start_list: Màu sắc bắt đầu
    :param stop_list: Màu sắc kết thúc
    :param is_horizontal_list: Danh sách bool > 3 item để xác định xem có bao nhiêu vùng gradient được tạo
    :return: Ảnh gradient
    """
    result = np.zeros((height, width, len(start_list)), dtype=np.float32)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def choice_background(use_color=True, width=300, heigth=30):
    """
    Tạo nền với màu sắc hoặc dùng ảnh từ thư mục chỉ định
    :param width:
    :param heigth:
    :param use_color: Bool - True ( Mặc định ): Tạo nền với màu - False: Sử dụng ảnh từ thư mục chỉ định
    :return: Ảnh nền <class 'PIL.Image.Image'>
    """
    try:
        if use_color:
            if bool(random.getrandbits(1)):
                L = Image.new('L', (width, heigth), random.randint(0, 255))
                RGB = Image.merge('RGB', (L, L, L))
                return RGB
            else:
                start_color = random_color(hex_code=False)
                end_color = random_color(hex_code=False)
                list_horizontal = sorted([bool(random.getrandbits(1)) for _ in range(random.randint(4, 7))],
                                         reverse=True)
                array_gradient = get_gradient_3d(width, heigth, start_color, end_color, list_horizontal)
                gradient = Image.fromarray(np.uint8(array_gradient))
                return gradient
        else:
            path = random.choice(list(Path('backgrounds').rglob('*.[jp][pn]*')))
            background_image = Image.open(path.as_posix())
            if background_image.size[0] > width and background_image.size[1] > heigth:
                left = random.randint(0, background_image.size[0] - width)
                right = left + width
                top = random.randint(0, background_image.size[1] - heigth)
                bottom = top + heigth
                return background_image.crop((left, top, right, bottom))
            return background_image.resize((width, heigth))
    except Exception as e:
        print_error('choice_background', e)




# def change_view(img):
#     imgs = cv2.copyMakeBorder(np.array(img), 100,100,100,100,cv2.BORDER_CONSTANT, None, value = 0)   
#     rows, cols, ch = imgs.shape
#     x_r = random.randint(-20, 20)
#     y_r = random.randint(-30, 30)
#     x_r1 = random.randint(-30, 30)
#     y_r1 = random.randint(-30, 30)
#     pts1 = np.float32([[100,100], [100+img.size[0],100], [100,100+img.size[1]], [100+img.size[0],100+img.size[1]]])
#     pts2 = np.float32([[100+random.randint(-10, 10),100+random.randint(-10, 10)], [random.randint(-10, 10)+100+img.size[0],100+random.randint(-10, 10)], [random.randint(-10, 10)+100,100+img.size[1]+random.randint(-10, 10)], [random.randint(-10, 10)+100+img.size[0],100+img.size[1]+random.randint(-10, 10)]])
#     # pts2 = np.float32([[0 + random.randint(-90, 90), 0], [cols + random.randint(-90, 90), 0], [0, rows], [cols, rows]])
#     # # pts2 = np.float32([[0 + 90, 0], [cols + -45, 0], [0, rows], [cols, rows]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     dst = cv2.warpPerspective(imgs, M, (cols + 70, rows + 70), flags=cv2.INTER_LINEAR)
#     gray = cv2.cvtColor(dst.copy(), cv2.COLOR_BGR2GRAY)
#     edged = cv2.Canny(gray, 30, 200)
#     contours, hierarchy = cv2.findContours(edged, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     # cv2.drawContours(dst, contours, -1, (255, 255, 255), 1)
#     x = [i for contour in contours for contou in contour for (i, j) in contou]
#     y = [j for contour in contours for contou in contour for (i, j) in contou]
#     (x1, x2, y1, y2) = (min(x), max(x), min(y), max(y))
#     dst = cv2.resize(dst[y1:y2,x1:x2], (90, 35))
#     # cv2.imshow('kc',dst)
#     # cv2.waitKey()
#     return dst
def change_view(img):
    img = np.array(img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[0 + random.randint(-20, 20), 0], [cols + random.randint(-20, 20), 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst


def insert_kitu(number):
    b= ''
    for i, kitu in enumerate(number):
        b += kitu
        if i !=len(number)-1:
            b += ' '
    return b


def invert_color_image(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_image = ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted_image.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
    else:
        result = ImageOps.invert(image)
    return result


def curved_image(img):
    ''' Bẻ cong ảnh theo trái or phải or giữa, trên xuống or dưới lên theo random, L là màu backgroud'''
    k = random.randint(700, 900)
    l = random.choice([-1, 1])
    img = np.array(img)
    rows, cols = img.shape[:2]
    img_output = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(0.0 * math.sin(2 * 3.14 * i / 350))
            offset_y = l * int(13.0 * math.cos(2 * 3.14 * j / k))
            if i + offset_y < rows and j + offset_x < cols:
                img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
            else:
                img_output[i, j] = 0
    return Image.fromarray(img_output)
    

# def sp_noise(image,prob):
#     '''
#     Add salt and pepper noise to image
#     prob: Probability of the noise
#     '''
#     output = np.zeros(image.shape,np.uint8)
#     thres = 1 - prob 
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

count = 0
no_folder = 84
count_all = 0
def load_fonts():
    """
        Load all fonts in the fonts directories
    """
    return [os.path.join('fonts/', font) for font in os.listdir('fonts/')]

def generator(line, fonts, isNumber=False):
    global count, count_all, no_folder
    char = string.ascii_letters + string.digits
    line = line.strip().lower()
    if not isNumber:
        if bool(random.getrandbits(1)):
            line = line.capitalize()
        if bool(random.getrandbits(1)):
            line = line.title()
        if bool(random.getrandbits(1)):
            line = line.upper()

    # for i in range(15):
    fonts = load_fonts()
    font = random.choice(fonts)

    image = ComputerTextGenerator.generate(line, font, '#0f0f0f', 25, orientation = 0, space_width =1)
    random_angle = random.randint(0-20, 20)
    rotated_img = image.rotate(random_angle, expand=1)

    distorsion_orientation =random.randint(0, 2)
    distorsion_type = 1#random.randint(0, 3)
    if distorsion_type == 0:
        distorted_img = rotated_img # Mind = blown
    elif distorsion_type == 1:
        distorted_img = DistorsionGenerator.sin(
            rotated_img,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
        )
    elif distorsion_type == 2:
        distorted_img = DistorsionGenerator.cos(
            rotated_img,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
        )
    elif distorsion_type == 3:
        distorted_img = DistorsionGenerator.random(
            rotated_img,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
        )
    size = 45
    # new_width = int(float(distorted_img.size[0] + 10) * (float(size) / float(distorted_img.size[1] + 10)))
    resized_img = distorted_img#.resize((new_width, distorted_img.size[1]), Image.ANTIALIAS)
    background_width =  distorted_img.size[0] + 10
    background_height = distorted_img.size[1] +10


    background_type = random.randint(0,3)
    if background_type == 0:
        background = BackgroundGenerator.gaussian_noise(background_height, background_width)
    elif background_type == 1:
        background = BackgroundGenerator.plain_white(background_height, background_width)
    elif background_type == 2:
        background = BackgroundGenerator.quasicrystal(background_height, background_width)
    elif background_type == 3:
        background = BackgroundGenerator.picture(background_height, background_width)


    new_text_width, _ = resized_img.size
    alignment = random.randint(0,2)
    if alignment == 0:
        background.paste(resized_img, (5, 5), resized_img)
    elif alignment == 1:
        background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), 5), resized_img)
    elif alignment == 2:
        background.paste(resized_img, (background_width - new_text_width - 5, 5), resized_img)
    # fig2 = plt.figure(figsize=(9, 5))
    # (ax1) = fig2.subplots(1)
    # ax1.imshow(background, cmap='gray')
    # ax1.axis('off')
    # plt.show()
    
    random_name = uuid.uuid4()    
    output_path = Path(f'data/images/{no_folder}')
    output_path.mkdir(parents=True, exist_ok=True)
    background.save(output_path.joinpath(f'{random_name}.png').as_posix(), 'PNG')
    with open('data/line_annotation.txt', 'a', encoding='utf-8') as f:
        f.write(f'{no_folder}/{random_name}.png\t{line}\n')
    count += 1
    count_all += 1
    print(count_all, end='\r')
    with open('log.txt', 'w') as f:
        f.write('Hinh da tao ' + str(count_all) + '\nSo luong folder ' + str(no_folder))
    if count > 199999:
        no_folder += 1
        count = 0


def run_gen(lines=None, fonts=None, gen_word=False):
    skip_word = r"'- !#$%&()*,./:;?@[\]^_|~+<=>«»0123456789" + '"'
    if fonts is None:
        fonts = []
    print(lines)
    for i, line in enumerate(lines):
        # if i > 5:
        #     break
        try:
            generator(line, fonts, isNumber=True)
        except Exception as e:
            raise e
        

if __name__ == '__main__':
    fonts = get_font('fonts')
    print_info('generation')
    # data = ["{:04n}".format(i) for i in range(9999)]#random.randint(1,9999)
    # for i in range(9999):
    #     data.append("{:04n}".format(i))
    #     data.append("{:04n}".format(i))
    data = open('data.txt', 'r', encoding='utf-8').read().strip().split('\n')[:1000]
    # print(data)
    np.random.shuffle(data)
    # print(data)
    run_gen(data, fonts, True)
    print()
