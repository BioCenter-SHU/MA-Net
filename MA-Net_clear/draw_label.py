import os
import cv2
from importlib_metadata import files

file_path = '/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/Mixed/233/'
img_name = os.listdir(file_path)
rects = []
with open('/YOUR_PATH/Clssification/DuoChiDu/data/233_gt.txt','r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        data = line.split(',')
        data = [int(x) for x in data]
        rects.append(data)

assert(len(img_name) == len(rects))

for i in range(len(img_name)):
    img = cv2.imread(file_path + img_name[i])
    rect = rects[i]
    img_rect = cv2.rectangle(img=img, pt1=(rect[2], rect[3]), pt2=(rect[2]+rect[4], rect[3]+rect[5]), color=(0,0,255))
    cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/draw_rect/233_det/{}.png".format(i), img_rect)
    # cv2.imshow("img", img_rect)
    # cv2.waitKey(1)


# img_15 = cv2.imread("16.jpg")
# img_rect_15 = cv2.rectangle(img=img_15, pt1=(308,239), pt2=(308+72, 239+25), color=(0,0,255))
# cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/233_15.png", img_rect_15)

# img_19 = cv2.imread("/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/Mixed/233/20.jpg")
# img_rect_19 = cv2.rectangle(img=img_19, pt1=(304,239), pt2=(304+64, 239+24), color=(0,0,255))
# cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/233_19.png", img_rect_19)

# img_31 = cv2.imread("/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/Mixed/233/31.jpg")
# img_rect_31 = cv2.rectangle(img=img_31, pt1=(287,227), pt2=(287+114, 227+31), color=(0,0,255))
# cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/233_31.png", img_rect_31)

# img_95 = cv2.imread("/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/Mixed/233/95.jpg")
# img_rect_95 = cv2.rectangle(img=img_95, pt1=(264,224), pt2=(264+64, 224+24), color=(0,0,255))
# cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/233_95.png", img_rect_95)

# img_107 = cv2.imread("/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/Mixed/233/107.jpg")
# img_rect_107 = cv2.rectangle(img=img_107, pt1=(278,223), pt2=(278+66, 223+23), color=(0,0,255))
# cv2.imwrite("/YOUR_PATH/Clssification/DuoChiDu/visualization/233_107.png", img_rect_107)