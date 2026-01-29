#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/2/19 16:39
@File:anycast_tcp_sever.py
@Desc:服务端的写法
"""
import socket
import time
import select
from threading import Thread
from utils import get_ip_addr



def sever_task(client,addr):
    print(f'接收到来自{addr}的连接')
    # 接收大数据包
    total_data = b''
    while True:
        # print('########################################')
        data = client.recv(1024)  # 每次接收1kB
        # print(f'接收到来自{addr}的数据{data}')
        total_data += data
        if b'\n' in total_data:
            receive_time = time.time()
            break

    print(f'接收到{len(total_data)}字节的数据包')
    # 发送响应
    massage = f'Received large packet:{receive_time}'
    client.send(massage.encode('utf-8'))
    client.close()
    print(f'与{addr}的连接已关闭')


# 启动服务器
def start_server(ip,port):
    server = socket.socket()
    server.bind((ip,port))
    server.listen(5)
    receive_time = time.time()

    print(f'服务器已启动，正在监听{ip}：{port}。。。')
    while True:
        client,addr = server.accept()
        server_thread = Thread(target=sever_task,args=(client,addr))  # 通过线程的方式实现服务端可以同时服务多个用户
        server_thread.start()



if __name__ == '__main__':
    # 定义服务端地址和端口
    ip = get_ip_addr('sta') if get_ip_addr('sta') else get_ip_addr('h')
    port = 5001
    start_server(ip,port)
