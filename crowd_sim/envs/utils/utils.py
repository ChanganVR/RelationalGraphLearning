import numpy as np
from collections import deque


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


class VTree(object):
    def __init__(self, state, current_level, width):
        self.state = state
        self.state_value_d_1 = None
        self.width = width
        self.current_level = current_level
        self.one_step_trajs_est = dict()
        self.children = dict()

    def add_child(self, action, state_est, width):
        self.children[action] = VTree(state_est, self.current_level + 1, width)

    def add_one_step_trajs(self, action, reward_est, state_est):
        self.one_step_trajs_est[action] = list()
        self.one_step_trajs_est[action].append(reward_est)
        self.one_step_trajs_est[action].append(state_est)


def bread_first_search_vtree(vtree):
    queue = deque([])
    vtree_list = []
    queue.append(vtree)
    while len(queue) > 0:
        tovisit = queue.popleft()
        for a, child in tovisit.children.items():
            queue.append(child)
        vtree_list.append(tovisit)
    return vtree_list


def print_vtree(vtree, action_space):
    vtree_list = bread_first_search_vtree(vtree)
    vtree_level = dict()
    for v in vtree_list:
        level = v.current_level
        vtree_level.setdefault(level, []).append(v)
    for level, vtrees in vtree_level.items():
        print('going to deal with trees in level_' + str(level) + ' :')
        width_index = 0
        for v in vtrees:
            width_index += 1

            for a, traj in v.one_step_trajs_est.items():
                a_index = action_space.index(a)
                r_level_a = traj[0]
                print('(a_' + str(a_index) + ', r: ' + str(r_level_a) + ') ', end="")

            if v.one_step_trajs_est != dict():
                print(' ')

            for a, child in v.children.items():
                a_index = action_space.index(a)
                print('at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' going to search deeper from a_' + str(a_index)+ ' with reward_' + str(v.one_step_trajs_est[a][0]) +' ')

            if v.state_value_d_1 != None:
                print('at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' the state_value_d_1 is: ' + str(v.state_value_d_1))
            else:
                print(' ')
        print('========================================================')
