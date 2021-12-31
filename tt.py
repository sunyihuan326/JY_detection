# -*- coding: utf-8 -*-
# @Time    : 2021/10/20
# @Author  : sunyihuan
# @File    : tt.py

import threading
import time
from queue import Queue


# print(threading.active_count())  # 当前几个线程
# print(threading.enumerate())  # 当前线程名称
# print(threading.current_thread())  # 当前运行的线程


def thresd_job():
    print("T1 start!!!\n")
    for i in range(10):
        time.sleep(0.1)
    print("This is :", threading.current_thread())


q = Queue()


def job(L):
    for i in range(len(L)):
        L[i] = L[i] ** 2
    q.put(L)


def main():
    add_thread = threading.Thread(target=thresd_job, name="T1")
    add_thread.start()
    add_thread.join()  # add_thread运行完成后执行后面语句

    print("all done\n")

    threads = []
    data = [[1, 2, 3], [4, 5, 6], [3, 5, 3], [9, 0, 4]]
    for i in range(4):
        t = threading.Thread(target=job, args=data[i])
        t.start()
        threads.append(t)
    for th in threads:
        th.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)


if __name__ == "__main__":
    main()

#
# exitFlag = 0
#
# threadLock = threading.Lock()
#
#
# class myThread(threading.Thread):
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#
#     def run(self):
#         print("开始线程：" + self.name)
#         threadLock.acquire()
#         print_time(self.name, self.counter, 5)
#         print("退出线程：" + self.name)
#         threadLock.release()
#
#
# def print_time(threadName, delay, counter):
#     while counter:
#         if exitFlag:
#             threadName.exit()
#         time.sleep(delay)
#         print("%s: %s" % (threadName, time.ctime(time.time())))
#         counter -= 1
#
#
# threads = []
# # 创建新线程
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
#
# # 开启新线程
# thread1.start()
# thread2.start()
#
# threads.append(thread1)
# threads.append(thread2)
# for t in threads:
#     t.join()
#
# print("退出主线程")


import tkinter
from tkinter import messagebox
from tkinter import simpledialog