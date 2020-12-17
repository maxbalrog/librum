import numpy as np
import matplotlib.pyplot as plt

import cv2

import skimage
from skimage import feature
from skimage.transform import rotate
from skimage import transform

import tensorflow as tf
import torch

import glob
import os

def plot_img(img):
    is_gray = len(img.shape) == 2
    plt.figure(dpi=200)
    if not is_gray:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def preprocess_img(img, resc_factor=3, kernel=(5,5)):
    h, w, _ = img.shape
    new_shape = (w//resc_factor, h//resc_factor)
    img_resc = cv2.resize(img, new_shape)

    imgray = cv2.cvtColor(img_resc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray,kernel,0)

    return img_resc, blur

def filter_rho(rho, theta, thr_dist=40):
    lines = [(abs(rho),rho,theta) for rho,theta in zip(rho,theta)]
    lines = sorted(lines, key=lambda x: x[0])

    lines_new = []
    x0_prev = -1e6
    for i,line in enumerate(lines):
        rho_loc, theta_loc = line[1:]
        x0_loc = rho_loc * np.cos(theta_loc)
        if x0_loc - x0_prev < thr_dist:
            continue
        else:
            x0_prev = x0_loc
            lines_new.append((line[1],line[2]))

    return lines_new


def filter_lines(rho, theta, n_bins=100, thr_dist=40, thr_angle=4):
    rho_ = np.array(rho)
    theta_ = np.array(theta)
    theta_ = theta_ / np.pi * 180

    #find dominant direction
    n, bins, _ = plt.hist(theta_, n_bins);
    dtheta = bins[1] - bins[0]
    print('dtheta: {}'.format(dtheta))

    n_conv = np.zeros_like(n)
    for i in range(1,n_bins-1):
        n_conv[i] += np.sum(n[i-1:i+2])
    n_conv[0] += n[-1] + np.sum(n[:2])

    idx_max = np.argmax(n_conv)
    theta_max = (bins[idx_max] + bins[idx_max+1]) / 2
    print('theta_max: {}'.format(theta_max))

    #filter out other lines
    idx_theta = (abs(theta_-theta_max) < thr_angle) | (abs(180-theta_-theta_max) < thr_angle)
    theta_new = theta_[idx_theta] / 180 * np.pi
    rho_new = rho_[idx_theta]

    lines = filter_rho(rho_new, theta_new, thr_dist=thr_dist)


    if 90-theta_max < 0:
        theta_max = 180 - theta_max
    print(theta_max)

    return lines, theta_max


def find_lines(img, sigma=3, thr=20):
    edges = feature.canny(img, sigma=sigma, low_threshold=thr)
    lines = cv2.HoughLines(np.uint8(edges), 1, np.pi / 180, 160)

    rho_list = []
    theta_list = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            rho_list.append(rho)
            theta_list.append(theta)

    return rho_list, theta_list


def rotate_img(img, thr=20, resc_factor=3, kernel=(5,5)):
    img_resc, imgray = preprocess_img(img, resc_factor=resc_factor, kernel=kernel)

    rho_list, theta_list = find_lines(imgray, thr=thr)

    lines_filtered, theta_max = filter_lines(rho_list, theta_list)

    #rotate image
    img_rotated = rotate(img_resc.astype(float), -theta_max)
    img_rotated = img_rotated.astype(np.uint8)
    img_rotated_gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

    rho, theta = find_lines(img_rotated_gray, thr=thr)

    thr_dist = img.shape[1] // 30
    lines_rotated, theta_rot_max = filter_lines(rho, theta, thr_dist)

    return img_resc, img_rotated, lines_filtered, lines_rotated


def calc_pt(line, img):
    h,w = img.shape[:2]
    rho, theta = line

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x_top = int(rho / a)
    x_top = np.clip(x_top, 0, w)
    y_top = 0

    x_bot = int((rho - h*b) / a)
    x_bot = np.clip(x_bot, 0, w)
    y_bot = h

    return [x_top,y_top], [x_bot,y_bot]


def calc_side_pts(line_1, line_2, img):
    h,w = img.shape[:2]
    pt_left_top, pt_left_bot = calc_pt(line_1, img)
    pt_right_top, pt_right_bot = calc_pt(line_2, img)

    side_short = pt_right_bot[0] - pt_left_bot[0]
    side_long = h

    pts = np.array([pt_left_top, pt_right_top, pt_right_bot, pt_left_bot], dtype=np.float32)

    return side_short, side_long, pts


def extract_books_warped(img, lines):
    h, w = img.shape[:2]
    books = []

    i = 0
    j = 1
    img1 = img.copy()
    while i < len(lines)-1:
        side_short, side_long, pts = calc_side_pts(lines[i], lines[i+1], img)
        dst = np.array([[0, 0], [side_short-1, 0], [side_short-1, side_long-1], [0, side_long-1]], dtype=np.float32)

        m = cv2.getPerspectiveTransform(pts, dst)
        book = cv2.warpPerspective(img1, m, (int(side_short), int(side_long)))
        books.append(book)

        i += 1

    return books


def extract_books(img, lines):
    h, w = img.shape[:2]
    books = []

    i = 0
    while i < len(lines)-1:
        rho, theta = lines[i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        l_star = h

        x1 = int(x0 + l_star*(-b))
        x2 = int(x0 - l_star*(-b))
        x_left = (x1+x2) // 2

        rho, theta = lines[i+1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + l_star*(-b))
        x2 = int(x0 - l_star*(-b))
        x_right = (x1+x2) // 2

        books.append(img[:h,x_left:x_right,:])

        i += 1

    return books
