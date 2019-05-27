'''
methods for preprocessing grand central station datasets offline
and for simulation in crowd_sim.py
'''
import scipy.io as sio
import numpy as np
import random
import os
import sys
import json
import csv


def position2annotation(p):
    panel_scale = 50
    panel_width = 1920
    panel_height = 1080
    px = p[0]
    py = p[1]
    ax = (px * panel_scale / panel_width + 0.5) * panel_width
    ay = (py * panel_scale / panel_height + 0.5) * panel_height
    return ax, ay


def annotation2position(a):
    panel_scale = 50
    panel_width = 1920
    panel_height = 1080
    ax = a[0]
    ay = a[1]
    px = (ax / panel_width - 0.5) * panel_width / panel_scale
    py = (ay / panel_height - 0.5) * panel_height / panel_scale
    return px, py


def count_total_human(human_ids):
    human_set = set(human_ids[0])
    for h_ids in human_ids:
        human_set = human_set | set(h_ids)


def count_total_human_from_f(f, start_t, end_t):
    p_set = set(f[start_t])
    for t in range(start_t, end_t + 1):
        p_set = p_set | set(f[t])
    return p_set


def if_f_p_accord(f_data_list, p_data_list):
    newf = [[] for i in range(7000)]
    for i in range(len(p_data_list)):
        p = p_data_list[i]
        tlist = [int(t) for t in list(p.keys())]
        for t in tlist:
            newf[t].append(i)
    print(f_data_list == newf[:5000])


def make_new_f(p_data_list):
    newf = [[] for i in range(7000)]
    for i in range(len(p_data_list)):
        p = p_data_list[i]
        tlist = [int(t) for t in list(p.keys())]
        for t in tlist:
            newf[t].append(i)
    return newf


def check_continue(a):
    b = [i for i in range(min(a), min(a) + len(a))]
    return a == b


def add_positions(tra):
    newtra = {}
    tlist = [int(t) for t in list(tra.keys())]
    newtlist = [i for i in range(min(tlist), max(tlist) + 1)]
    for t in newtlist:
        if t in tlist:
            newtra[str(t)] = tra[str(t)]
        else:
            newtra[str(t)] = newtra[str(t - 1)]
    return newtra


def create_GC_metadata(raw_data_path, meta_data_path):
    """
    create data from downloaded raw data to meta data ( a data structure to read easily)
    :param raw_data_path: downloaded raw data path
    :param meta_data_path: meta data path
    :return:
    """

    GC_IMAGE_WIDTH = 1920
    GC_IMAGE_HEIGHT = 1080

    dir_list = sorted(os.listdir(raw_data_path))
    p_num = len(dir_list)

    # + 1 because raw GC txt start from 1,just add a fake person whose pid = 0
    p_data_list = [{} for _ in range(p_num + 1)]

    # fill p_data
    max_t = 0
    for dir_name in dir_list:

        person_trajectory_txt_path = os.path.join(raw_data_path, dir_name)
        pid = int(dir_name.replace('.txt', ''))
        if pid == 1:
            print('check')
        with open(person_trajectory_txt_path, 'r') as f:
            trajectory_list = f.read().split()
            for i in range(len(trajectory_list) // 3):
                x = int(trajectory_list[3 * i]) / GC_IMAGE_WIDTH
                y = int(trajectory_list[3 * i + 1]) / GC_IMAGE_HEIGHT
                t = int(trajectory_list[3 * i + 2]) // 20
                max_t = max(max_t, t)
                p_data_list[pid][t] = (x, y)

    # fill f_data
    f_data_list = [[] for _ in range(max_t + 1)]
    for pid, p_data in enumerate(p_data_list):
        for t in p_data.keys():
            f_data_list[t].append(pid)

    # show some message
    print('pedestrian_data_list size: ', len(p_data_list))
    print('frame_data_list size: ', len(f_data_list))
    print('max num of person in one frame', count_frame_p_num(f_data_list))

    with open(meta_data_path, 'w') as f:
        json.dump({'frame_data_list': f_data_list, 'pedestrian_data_list': p_data_list}, f)

    print('create %s successfully!' % meta_data_path)


def count_frame_p_num(f_data_list):
    p_num_list = []
    for f in f_data_list:
        p_num_list.append(len(f))
    return max(p_num_list)


def main():
    GC_raw_data_path = 'data/GC/Annotation'
    GC_meta_data_path = 'data/GC_meta_data.json'
    GC_train_test_data_path = 'data/GC.npz'

    create_GC_metadata(GC_raw_data_path, GC_meta_data_path)
    create_GC_train_test_data(GC_meta_data_path, GC_train_test_data_path)


if __name__ == '__main__':
    main()
