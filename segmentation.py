import cv2
import mediapipe as mp
import numpy as np

BG_COLOR = (150, 0, 150)

selfie_segmentation = mp.solutions.selfie_segmentation


def fit_img(img):
    """fit the image to an specified size"""
    target_width = 1080
    proper_height = 1080
    if width > target_width or height > proper_height:
        print('entro al if')
        aspect_ratio = width / height
        target_height = int(target_width / aspect_ratio)
        print(f'Original size(w, h): {(width, height)}\nOriginal aspect ratio: {aspect_ratio}')
        img = cv2.resize(img, (target_width, target_height))

        return img
    else:
        return img


with selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:

    image = cv2.imread('imgs/6.jpg')
    bg_content = cv2.imread('bgs/1.jpg')
    # Get the image size
    height, width = image.shape[:2]

    image = fit_img(image)

    # get a RGB img
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = selfie_segmentation.process(image_rgb)

    # normalization
    _, th = cv2.threshold(results.segmentation_mask, 0.65, 255,
                          cv2.THRESH_BINARY)

    print(th.dtype)
    th = th.astype(np.uint8)
    th = cv2.medianBlur(th, 13)
    th_inv = cv2.bitwise_not(th)

    # Background
    # bg_content = np.ones(image.shape, dtype=np.uint8)
    # bg_content[:] = BG_COLOR
    
    bg_content = fit_img(bg_content)
    bg = cv2.bitwise_and(bg_content, bg_content, mask=th_inv)

    # # Foreground
    fg = cv2.bitwise_and(image, image, mask=th)

    # # Background + Foreground
    output = cv2.add(bg, fg)

    cv2.imshow('image', image)
    # cv2.imshow('mask', results.segmentation_mask)
    # cv2.imshow('th', th)
    # cv2.imshow('th_inv', th_inv)
    # cv2.imshow('bg_content', bg_content)
    # cv2.imshow('bg', bg)
    # cv2.imshow('fg', fg)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.imwrite('output_image.jpg', output)
