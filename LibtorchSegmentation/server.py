# -*- coding=utf-8 -*-

"""
file: recv.py
socket service
"""
import socket
import threading
import sys
import struct
import numpy as np
import cv2
#import pyzed.sl as sl
import argparse

"""
fx = 530.2999877929688
fy = 530.0
cx = 658.239990234375
cy = 370.7825012207031
"""
fx = 528.8250122070312
fy = 528.4199829101562
cx = 634.2100219726562
cy = 357.22900390625

depth_w = 1280
depth_h = 720

def socket_service():
    #while 1:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 50000))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    
    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


def deal_data(conn, addr):
    debug = args.debug
    while (1):
        buf_size = 512*512*1  # 获取读的图片总长度
        if buf_size:
            buf = b""  # 代表bytes类型
            temp_buf = buf
            while (buf_size):  # 读取每一张图片的长度
                temp_buf = conn.recv(buf_size)
                buf_size -= len(temp_buf)
                buf += temp_buf  # 获取图片
        data = np.frombuffer(buf, dtype='uint8')  # 按uint8转换为图像矩阵
        image = data.reshape(512,512,1)


        buf_size = depth_w*depth_h*2  # 获取读的图片总长度
        if buf_size:
            buf = b""  # 代表bytes类型
            temp_buf = buf
            while (buf_size):  # 读取每一张图片的长度
                temp_buf = conn.recv(buf_size)
                buf_size -= len(temp_buf)
                buf += temp_buf  # 获取图片
        data = np.frombuffer(buf, dtype='uint16')  # 按uint8转换为图像矩阵
        depth = data.reshape(depth_h,depth_w,1)
        depth = cv2.resize(depth, (1280,720))


        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (1280,720))
        image[666:,50:1250] = 255
        # image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 图像解码
        img_expand = np.zeros((image.shape[0],image.shape[1]+2,3)).astype(np.uint8)
        img_vis = np.zeros((image.shape[0],image.shape[1]+2,3)).astype(np.uint8)
        img_expand[:, 1:-1, :] = image
        binary = img_expand[:,:,0]
        final = cv2.Canny(binary, 80, 255)
        contours, hierchary = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_filter = []
        max_length = 0
        index = -1
        for idx, contour in enumerate(contours):
            if contour.shape[0] > max_length:
                max_length = contour.shape[0]
                index = idx
        contours_filter.append(contours[index])
        final = cv2.drawContours(img_vis, contours_filter, -1, (0, 0, 255), 2)
        final = final[:,:,-1]
        if debug:
            cv2.imwrite('temp_depth.png', depth)
            cv2.imwrite('temp.jpg', final)	
        """
        edge_array = contours_filter[0].reshape(-1,2)
        img_x = edge_array[:, 0]
        img_x = np.clip(img_x, 1, 1280) - 1
        img_y = edge_array[:, 1]
        img_yx = np.dstack((img_y,img_x)).reshape(-1,2)
        """
        s = 4
        final = cv2.resize(final, (int(1280/s),int(720/s)))
        edge_array = np.where(final > 1)
        img_x = edge_array[1] * s
        img_x = np.clip(img_x, 1, 1280) - 1
        img_y = edge_array[0] * s
        img_yx = np.dstack((img_y,img_x)).reshape(-1,2)
        img_z = []
        for y,x in img_yx:
            img_z.append(depth[y,x])
        img_z = np.array(img_z)
        z=img_z
        x=(img_x-cx)*z/fx
        y=(img_y-cy)*z/fy
        xyz=np.dstack((x,y,z)).reshape(-1,3).reshape(-1)
        print(depth[300,600])
        #print(img_y.shape, img_x)		
        print("processed one image!")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    socket_service()
