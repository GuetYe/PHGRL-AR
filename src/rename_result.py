#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/5/8 22:00
@File:rename_result.py
@Desc:****************
"""
# 清空历史模型数据
import shutil
import os
import setting
import time
import sys
result_path = setting.result_path
# path_list = [setting.model_path,setting.save_figure_path,setting.save_data_path]
path_list = [setting.result_path]
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

def rename_data(tag):
    for path in path_list:
        original_path = os.path.normpath(path)
        if os.path.exists(original_path):
            # 复制文件
            now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            renamed_path = f'{original_path}_{tag}_{now_time}'
            os.rename(original_path,renamed_path)
            print(f"Renamed: {original_path} -> {renamed_path}")

            # 重新创建和原文件夹
            create_empty_structure(renamed_path,original_path)
            print(f"Recreated empty folder structure: {original_path}")
        else:
            # 如果路径不存在，直接创建
            print(f"Path does not exist: {original_path}")
            os.makedirs(original_path)
            print(f"Created folder: {original_path}")

def create_empty_structure(renamed_path,original_path):
    """
        根据源文件夹的目录结构，在目标路径创建相同的空文件夹。
    """
    for root, dirs, files in os.walk(renamed_path):
        # 计算相对路径
        relative_path = os.path.relpath(root, renamed_path)
        # 构建目标路径
        new_dir = os.path.join(original_path, relative_path)
        # 创建空文件夹
        os.makedirs(new_dir, exist_ok=True)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        tag = ''
    else:
        tag = sys.argv[1]
    rename_data(tag)
