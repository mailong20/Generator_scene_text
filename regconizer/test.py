import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
import random

def get_transformed_image(src, dst, img):
    # calculate the tranformation
    mat = cv2.getPerspectiveTransform(src.astype("float32"), dst.astype("float32"))
    
        
    # new source: image corners
    corners = np.array([
                    [0, img.size[0]],
                    [0, 0],
                    [img.size[1], 0],
                    [img.size[1], img.size[0]]
                ])

    # Transform the corners of the image
    corners_tranformed = cv2.perspectiveTransform(
                                  np.array([corners.astype("float32")]), mat)

    # These tranformed corners seems completely wrong/inverted x-axis 
    print(corners_tranformed)
    
    x_mn = math.ceil(min(corners_tranformed[0].T[0]))
    y_mn = math.ceil(min(corners_tranformed[0].T[1]))

    x_mx = math.ceil(max(corners_tranformed[0].T[0]))
    y_mx = math.ceil(max(corners_tranformed[0].T[1]))

    width = x_mx - x_mn
    height = y_mx - y_mn

    analogy = height/1000
    n_height = height/analogy
    n_width = width/analogy


    dst2 = corners_tranformed
    dst2 -= np.array([x_mn, y_mn])
    dst2 = dst2/analogy 

    mat2 = cv2.getPerspectiveTransform(corners.astype("float32"),
                                       dst2.astype("float32"))


    img_warp = Image.fromarray((
        cv2.warpPerspective(np.array(image),
                            mat2,
                            (int(n_width),
                            int(n_height)))))
    return img_warp


# image coordingates


rows, cols, ch = (500, 500, 3)
image = Image.new('RGB', (rows, cols))
src = np.float32([[0, 0], [cols - 1, 0],
                [0, rows - 1], [cols - 1, rows - 1]])
dst = np.float32([[0 + random.randint(-90, 90), 0], [cols + random.randint(-90, 90), 0], [0, rows], [cols, rows]])

# Create the image

image.paste( (200,200,200), [0,0,image.size[0],image.size[1]])
draw = ImageDraw.Draw(image)
draw.line(((src[0][0],src[0][1]),(src[1][0],src[1][1]), (src[2][0],src[2][1]),(src[3][0],src[3][1]), (src[0][0],src[0][1])), width=4, fill="blue")
#image.show()

warped =  get_transformed_image(src, dst, image)
warped.show()