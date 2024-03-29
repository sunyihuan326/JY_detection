#!/usr/bin/env python
# -*- coding=utf-8 -*-
import socket
import threading
import time
import sys
import os
import struct

ip_addr = "192.168.3.118"
max_file_size = 4096


def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip_addr, 8888))  # 这里换上自己的ip和端口
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print("Waiting...")

    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()


def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))
    while 1:
        fileinfo_size = struct.calcsize('128sl')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            print(filename)
            fn = filename.strip(str.encode('\00'))
            new_filename = os.path.join(str.encode('./'), str.encode('new_') + fn)
            print('file new name is {0}, filesize if {1}'.format(new_filename, filesize))

            recvd_size = 0  # 定义已接收文件的大小
            fp = open(new_filename, 'wb')
            print("start receiving...")
            while not recvd_size == filesize:
                if filesize - recvd_size > max_file_size:
                    data = conn.recv(max_file_size)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print("end receive...")

        conn.close()
        break


if __name__ == '__main__':
    socket_service()

    # cv2.imread(new_filename)
    # cv2.namedWindow('camera_output', 0)
    # cv2.imshow('camera_output', new_filename)
    # cv2.waitKey(100)
