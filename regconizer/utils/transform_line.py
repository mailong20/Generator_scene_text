""" Created by MrBBS """
# 12/23/2021
# -*-encoding:utf-8-*-

import math

import cv2
import numpy as np

from .deslantImg import deslant_img


def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def get_box(textmap, linkmap):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    text_score_comb = np.clip(textmap + linkmap, 0, 1)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        # remove link area
        segmap[np.logical_and(linkmap == 1, textmap == 0)] = 0

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = (x - niter, x + w + niter + 1, y - niter, y + h + niter + 1)
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
        np_contours = np_temp.transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)
    return det, labels, mapper


def get_poly(boxes, labels, mapper):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = (
            int(np.linalg.norm(box[0] - box[1]) + 1),
            int(np.linalg.norm(box[1] - box[2]) + 1),
        )
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [
                    cp_section[seg_num][0] / num_sec,
                    cp_section[seg_num][1] / num_sec,
                ]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [
                cp_section[seg_num][0] + x,
                cp_section[seg_num][1] + cy,
            ]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh
        # is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = -math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (
                pp[2][1] - pp[1][1]
        ) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (
                pp[-3][1] - pp[-2][1]
        ) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break
        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def padd(poly, w_pad=0):
    half = len(poly) // 2
    max_temp = poly[half - 2]
    max_1 = poly[half - 1].copy()
    max_2 = poly[half].copy()
    min_temp = poly[1]
    min_1 = poly[0].copy()
    min_2 = poly[-1].copy()

    heigh_max = max(max_2[1], max_1[1]) - min(max_2[1], max_1[1])
    heigh_min = max(min_2[1], min_1[1]) - min(min_2[1], min_1[1])
    ac = ((((max_1[0] - min_2[0]) ** 2) + ((max_1[1] - min_2[1]) ** 2)) ** 0.5)
    if ac < 50 or heigh_max < 5 or heigh_min < 5:
        return None
    if ac > 0:
        w_pad = ac / 5

    angle_max = math.atan2(max_temp[1] - max_1[1], max_temp[0] - max_1[0])
    angle_min = math.atan2(min_temp[1] - min_1[1], min_temp[0] - min_1[0])

    y_min_vertical = int(w_pad * math.sin(angle_min))
    y_max_vertical = int(w_pad * math.sin(angle_max))

    max_vertical = [[max_1[0] + w_pad, max_1[1] + y_max_vertical], [max_2[0] + w_pad, max_2[1] + y_max_vertical]]
    min_vertical = [[min_1[0] - w_pad, min_1[1] + y_min_vertical], [min_2[0] - w_pad, min_2[1] + y_min_vertical]]
    poly = np.insert(poly, half, max_vertical, axis=0)
    poly = np.insert(poly, 0, min_vertical[0], axis=0)
    poly = np.append(poly, [min_vertical[1]], axis=0)
    poly = np.array([[p[0], p[1] + (max(heigh_max, heigh_min) // 3.5) if i > half + 1 else p[1] - (
            max(heigh_max, heigh_min) // 3.5)] for i, p in enumerate(poly)])
    return poly


def rectify_poly(img, poly):
    # Use Affine transform
    poly_padded = padd(poly)
    if poly_padded is not None:
        poly = poly_padded
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE  # , borderValue=color
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)


def straight_image(image):
    img = cv2.copyMakeBorder(image, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 39)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.dilate(threshold.copy(), kernel)
    text_map = threshold.copy()
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        centers.append([x + w // 2, y + h // 2])
    centers = sorted(centers, key=lambda x: x[0])
    for i in range(len(centers)):
        if i < len(centers) - 1:
            cv2.line(threshold, centers[i], centers[i + 1], (255, 255, 255), 3)
    threshold = cv2.dilate(threshold, kernel)
    threshold = cv2.erode(threshold, kernel)
    threshold = cv2.blur(threshold, (10, 10))
    boxes, labels, mapper = get_box(text_map, threshold)
    polys = get_poly(boxes, labels, mapper)
    if len(polys) > 0 and polys[0] is None:
        polys = boxes
    if len(polys) > 0:
        result = rectify_poly(img, polys[0])
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 39)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        threshold = cv2.dilate(threshold.copy(), kernel)
        idx = np.where(np.all(cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2BGR) == (255, 255, 255), 2))
        if not len(idx[0]) == 0 and not len(idx[1]) == 0:
            x1, y1, x2, y2 = idx[1].min(), idx[0].min(), idx[1].max(), idx[0].max()
            result = result[y1:y2, x1 - 5:x2]
        img = deslant_img(result).img
        return img


if __name__ == '__main__':
    from pathlib import Path
    import time

    imgs = Path(r'D:\SourceCode\bill_extract\curve_line').rglob('*.[jp][pn]*')
    start_time = time.time()
    for p in imgs:
        img = cv2.imread(p.as_posix())
        result = straight_image(img)
        # cv2.imshow('bb', img)
        # cv2.imshow('cc', result)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    print(time.time() - start_time)
