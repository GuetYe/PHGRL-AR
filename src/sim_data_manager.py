#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/7/28 22:18
@File:data_manager.py
@Desc:****************
"""
import os
import re
import time
from pathlib import Path
from datetime import datetime


def extract_prefix_and_number(filename):
    """
    解析文件名，提取前缀和末尾的数字编号
    :param filename:
    :return:
    """
    # 去掉扩展名
    name, ext = os.path.splitext(filename)
    # print(f'文件名{name},扩展名{ext}')

    # 匹配末尾数字
    m = re.search(r'\d+',name)
    # m = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}',filename)
    # print(f'配匹结果{m}')
    if not m:
        # m = re.search(r'\d+$', name)
        # if not m:
        #     return name, None
        return name, None

    number = m.group(0)
    # print(f'数字{number}')
    # timestamp = m.group(0)
    # time_format = "%Y-%m-%d-%H-%M-%S"

    prefix = name.split('_')[0]
    # prefix = name[:-len(timestamp)]
    # print(f'前缀{prefix}')
    # dt = datetime.strptime(timestamp,time_format)
    # times = dt.timestamp()

    return prefix, number

def keep_latest_files(root_dir):
    """
    遍历目录，针对同一目录同一前缀，保留最新时间的文件，删除其余文件
    :param root_dir:
    :return:
    """
    last_files = {}  # key:(dirpath,prefix) -> (max_time,file_path)
    logger_last_time = -1
    last_file = ''
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            prefix, number = extract_prefix_and_number(filename)
            print(prefix)
            if prefix is None:
                continue
            key = (dirpath,prefix)
            cur_path = Path(dirpath) / filename

            # 获取时间戳
            mod_time = os.path.getmtime(cur_path)
            time_tuple = time.localtime(mod_time)
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time_tuple)
            # print(f'{cur_path}最后修改时间：{formatted_time}')
            if 'logger' in prefix:
                if mod_time > logger_last_time:  # 当前文件比较新，保留当前文件，如果文件存在删除文件，更新文件
                    if last_file:
                        pass
                        # print(f'删除文件:{last_file}')
                    last_file = cur_path
                    logger_last_time = mod_time

                else: # 当前文件不新，删除文件
                    pass
                    # print(f'删除文件:{cur_path}')
            else:
                if key not in last_files or int(number) > int(last_files[key][0]):
                    last_files[key] = (number,formatted_time,cur_path)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            prefix,number = extract_prefix_and_number(filename)
            if prefix is None:
                continue
            key = (dirpath, prefix)
            cur_path = Path(dirpath) / filename
            mod_time, formatted_time, max_file_path = last_files.get(key,(None,None,None))
            if 'logger' in prefix:
                pass
            else:
                if cur_path != max_file_path:
                    print(f'删除旧文件：{cur_path}')
                    os.remove(cur_path)

    print(f'时间最新文件字典：{last_files}')


if __name__ == '__main__':
    file_name = './result/'
    # extract_prefix_and_number(file_name)
    keep_latest_files(file_name)

