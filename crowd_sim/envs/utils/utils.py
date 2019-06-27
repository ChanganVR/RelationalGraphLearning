import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
        self.q_value_d_n = dict()

    def add_child(self, action, state_est, width):
        self.children[action] = VTree(state_est, self.current_level + 1, width)

    def add_one_step_trajs(self, action, reward_est, next_state_est, next_state_value_d_1):
        self.one_step_trajs_est[action] = list()
        self.one_step_trajs_est[action].append(reward_est)
        self.one_step_trajs_est[action].append(next_state_est)
        self.one_step_trajs_est[action].append(next_state_value_d_1)


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


def print_vtree(vtree, action_space, robot, global_step, action_taken_index):
    import matplotlib.pyplot as plt
    precision = 4
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
            values_to_plot = []
            Q_to_plot = []
            rewards_to_plot = []
            Q_d_to_plot = []
            for a, traj in v.one_step_trajs_est.items():
                a_index = action_space.index(a)
                r_level_a = traj[0]
                next_state_value_d_1 = traj[2]
                values_to_plot.append(next_state_value_d_1)
                Q_to_plot.append(next_state_value_d_1 * robot.policy.get_normalized_gamma() + r_level_a)
                rewards_to_plot.append(r_level_a)
                if v.current_level < 1:
                    Q_d_to_plot.append(v.q_value_d_n[a])
                print('(a_' + str(a_index) + ', r: ' + str(round(r_level_a, precision)) + ', v(s_t+1): ' + str(round(next_state_value_d_1, precision)) + ') ', end="")
            if v.one_step_trajs_est != dict() and v.current_level < 1:
                plot_value_heatmap_title = 'global_step: '+ str(global_step) + ' level_' + str(level) + 'width_' + str(width_index) + '_next_state_value' + '_a_taken_' + str(action_taken_index) +\
                    str(' ') + str(np.array(values_to_plot).argsort()[::-1][:5].tolist())
                plot_vtree_heatmap(plot_value_heatmap_title, robot, values_to_plot, plt)

                plot_value_heatmap_title = 'global_step: '+ str(global_step) + ' level_' + str(level) + 'width_' + str(width_index) + '_Q' + '_a_taken_' + str(action_taken_index) +\
                    str(' ') + str(np.array(Q_to_plot).argsort()[::-1][:20].tolist())
                plot_vtree_heatmap(plot_value_heatmap_title, robot, Q_to_plot, plt)

                plot_value_heatmap_title = 'global_step: '+ str(global_step) + ' level_' + str(level) + 'width_' + str(width_index) + '_r' + '_a_taken_' + str(action_taken_index) +\
                    str(' ') + str(np.array(rewards_to_plot).argsort()[::-1][:5].tolist())
                plot_vtree_heatmap(plot_value_heatmap_title, robot, rewards_to_plot, plt)

                plot_value_heatmap_title = 'global_step: '+ str(global_step) + ' level_' + str(level) + 'width_' + str(width_index) + '_Q-d' + '_a_taken_' + str(action_taken_index) +\
                    str(' ') + str(np.array(Q_d_to_plot).argsort()[::-1][:5].tolist())
                plot_vtree_heatmap(plot_value_heatmap_title, robot, Q_d_to_plot, plt)

            if v.one_step_trajs_est != dict():
                print(' ')

            for a, child in v.children.items():
                a_index = action_space.index(a)
                print('at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' going to search deeper from a_' + str(a_index)+ ' with reward: ' + str(round(v.one_step_trajs_est[a][0], precision)) + \
                      ' and v(s_t+1): ' + str(round(v.one_step_trajs_est[a][2], precision)) + ' ')

            if v.state_value_d_1 != None:
                print('at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' the state_value_d_1 is: ' + str(round(v.state_value_d_1, precision)))
                print(' ')
        print('========================================================')
    plt.show()

def plot_vtree_heatmap(title, robot, values, plt):
    # when any key is pressed draw the action value plot

    fig, axis = plt.subplots()
    axis.set_title(title)

    speeds = [0] + robot.policy.speeds

    rotations = robot.policy.rotations.tolist() + [np.pi * 2]
    r, th = np.meshgrid(speeds, rotations)

    z = np.array(values[1:])

    if np.max(z) != np.min(z):
        z = (z - np.min(z)) / (np.max(z) - np.min(z))

    z = np.reshape(z, (robot.policy.rotation_samples, robot.policy.speed_samples))

    polar = plt.subplot(projection="polar")
    polar.tick_params(labelsize=16)
    polar.set_title(title)
    mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
    plt.plot(rotations, r, color='k', ls='none')
    plt.grid()
    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(mesh, cax=cbaxes)
    cbar.ax.tick_params(labelsize=16)



def plot_vtree(vtree, action_space, plt, global_step):
    count_line = 100
    line_height = 1.0 / count_line
    line_num = 2

    fig, ax = plt.subplots()
    ax.text(0, count_line * line_height, 'global time step: ' + str(global_step))
    count_line -= line_num

    vtree_list = bread_first_search_vtree(vtree)
    vtree_level = dict()
    for v in vtree_list:
        level = v.current_level
        vtree_level.setdefault(level, []).append(v)
    for level, vtrees in vtree_level.items():
        ax.text(0, count_line * line_height, 'going to deal with trees in level_' + str(level) + ' :')
        count_line -= line_num

        width_index = 0
        for v in vtrees:
            width_index += 1

            for a, traj in v.one_step_trajs_est.items():
                a_index = action_space.index(a)
                r_level_a = traj[0]
                ax.text(0, count_line * line_height, '(a_' + str(a_index) + ', r: ' + str(r_level_a) + ') ', end="")
                count_line -= line_num

            if v.one_step_trajs_est != dict():
                ax.text(0, count_line * line_height, ' ')
                count_line -= line_num

            for a, child in v.children.items():
                a_index = action_space.index(a)
                ax.text(0, count_line * line_height, 'at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' going to search deeper from a_' + str(a_index)+ ' with reward_' + str(v.one_step_trajs_est[a][0]) +' ')
                count_line -= line_num

            if v.state_value_d_1 != None:
                ax.text(0, count_line * line_height, 'at level ' + str(v.current_level) + ' width ' + str(width_index) + '-st' + ' the state_value_d_1 is: ' + str(v.state_value_d_1))
                count_line -= line_num
            else:
                ax.text(0, count_line * line_height, ' ')
                count_line -= line_num
        ax.text(0, count_line * line_height, '========================================================')
        count_line -= line_num
    plt.show()