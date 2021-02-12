import cv2 as cv
import numpy as np
import sys


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    #crop right
    elif not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


print("Please enter image paths from left image to right image for stitching")
print("After entering images please enter x to start stitching")
img_list = []
while True:
    img_name = input("Please enter image path:\n")
    if img_name == "x":
        break
    else:
        img = cv.imread(img_name)
        img_list.append(img)

img_right = img_list[len(img_list) - 1]
i = len(img_list) - 2
bf = cv.BFMatcher_create()
sift = cv.SIFT_create()
while i >= 0:
    img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    img_left = img_list[i]
    img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)

    kp_right, des_right = sift.detectAndCompute(img_right_gray, None)
    kp_left, des_left = sift.detectAndCompute(img_left_gray, None)

    matches = bf.knnMatch(des_right, des_left, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) >= MIN_MATCH_COUNT:
        src_points = np.float32([kp_right[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_points = np.float32([kp_left[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    else:
        print("Number of matches found " + str(len(good)) + " but need at least " + str(MIN_MATCH_COUNT) + " matches please try again")
        print("Closing Program...")
        sys.exit()

    warped_img = cv.warpPerspective(img_right, M, (img_right.shape[1] + img_left.shape[0], img_left.shape[0]))
    warped_img[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
    img_right = trim(warped_img)
    i = i - 1

img_stitched = img_right
img_stitched_filename = input("Please enter an image path to save the stitched image:\n")
cv.imwrite(img_stitched_filename, img_stitched)
