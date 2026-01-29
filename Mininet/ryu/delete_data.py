#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/5/3 19:26
@File:delete_data.py
@Desc:****************
"""
# 清空历史模型数据
import shutil
import os
import setting
path_list = ['./savepath/', './save_action/', './dst_node_pro_file/']
def delete_data():
    for path in path_list:
        new_model_path = os.path.normpath(path)
        if os.path.exists(new_model_path):
            shutil.rmtree(new_model_path)
            print(f"Deleted: {new_model_path}")
            os.makedirs(new_model_path)
            print(f"Recreated folder: {new_model_path}")
        else:
            print(f"Path does not exist: {new_model_path}")
            os.makedirs(new_model_path)
            print(f"Recreated folder: {new_model_path}")

if __name__ == '__main__':
    delete_data()