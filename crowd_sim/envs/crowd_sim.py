import logging
import math
import gym
import matplotlib.lines as mlines
import numpy as np
import random
import json
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.realsim_utils.GrandCentral import *

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.group_num = None
        self.group_size = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        #if human policy are from the real data
        self.f_data_list = None
        self.p_data_list = None
        self.t_start = 0

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.save_scene_dir = None
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        #for debug
        self.add_human = []
        self.delete_human = []
        self.total_group_size = 0
        self.hp_25 = {}
        self.ha_25 = {}
        self.phase = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.group_num = config.sim.group_num
        self.group_size = config.sim.group_size
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_human(self, human=None):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        if self.current_scenario == 'circle_crossing':

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, -px, -py, 0, 0, 0)


        elif self.current_scenario == 'square_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gx = np.random.random() * self.square_width * 0.5 * - sign
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def generate_group(self, phase='train', group_size=3, t_in_real=None, p_set=None):
        '''
        sps: start positions list
        gps: goal positions list
        '''
        if self.current_scenario == 'group_circle_crossing':
            group = [Human(self.config, 'humans') for i in range(group_size)]
            circle_radius = self.circle_radius
            human = group[0]
            door_distance = human.radius + self.discomfort_dist

            if self.count_el_pair % 2:
                el_type = 'crossing'
            else:
                el_type = 'crossing'

            def generate_entry_leave(circle_radius, phase, type = 'crossing'):
                if phase == 'train' or phase == 'val':
                    angle = np.random.random() * np.pi * (0.5) + np.pi * 0.5
                else:
                    a = self.case_counter[phase] / float(self.case_size[phase])
                    #a = np.random.random()
                    self.test_scene_seeds.append(a)
                    angle = a * np.pi * (0.5)
                # add some noise to simulate all the possible cases robot could meet with human

                if type == 'crossing':
                    '''
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    ex = circle_radius * np.cos(angle) + px_noise
                    ey = circle_radius * np.sin(angle) + py_noise
                    '''
                    ex = circle_radius * np.cos(angle)
                    ey = circle_radius * np.sin(angle)

                else:
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    px_noise = np.random.random() * 2

                    ex = (self.robot.get_start_position()[0] + sign *( 3 + px_noise ))
                    if sign == 1:
                        # the human and robot start from the same direction
                        ey = self.robot.get_start_position()[1]
                    else:
                        # the human and robot starts from the opposite direction
                        ey = self.robot.get_goal_position()[1]

                if type == 'crossing':
                    lx = -ex
                    ly = -ey
                elif type == 'perpendicular':
                    lx = ex
                    ly = -ey
                else:
                    raise NotImplementedError
                return ex, ey, lx, ly

            while True:
                ex, ey, lx, ly = generate_entry_leave(circle_radius, phase, el_type)
                if len(self.entries) > 0:
                    for i, e in enumerate(self.entries):
                        l = self.leaves[i]
                        e_collide = norm((e[0] - ex, e[1] - ey)) < door_distance
                        l_collide = norm((l[0] - lx, l[1] - ly)) < door_distance
                        collide = e_collide | l_collide
                        if collide:
                            break
                    r_s_collide = norm((ex - self.robot.get_start_position()[0], ey - self.robot.get_start_position()[1])) < self.robot.radius + human.radius + self.discomfort_dist * 10
                    r_g_collide = norm((lx - self.robot.get_goal_position()[0], ey - self.robot.get_goal_position()[1])) < self.robot.radius + human.radius + self.discomfort_dist * 10

                    collide = e_collide | l_collide | r_s_collide | r_g_collide
                else:
                    break
                if not collide:
                    break

            self.entries.append([ex, ey])
            self.leaves.append([lx, ly])

            self.count_el_pair += 1

            # assume perpendicular entry and horizontal leave
            # todo: add poisson distribution to model human enter time
            sps = [[ex, ey + i * (human.radius + self.discomfort_dist)] for i in range(-(group_size//2), (group_size//2)+1)]
            gps = [[lx + i * (human.radius + self.discomfort_dist), ly] for i in range(-(group_size//2), (group_size//2)+1)]

            # shuffle to avoid parallel path
            random.shuffle(sps)
            random.shuffle(gps)

            for i, human in enumerate(group):
                human.set(sps[i][0], sps[i][1], gps[i][0], gps[i][1], 0, 0, 0)

        elif self.current_scenario == 'realsim_GrandCentral':
            if p_set == None:
                p_set = self.f_data_list[t_in_real]
            group_size = len(p_set)
            self.total_group_size += group_size
            group = [Human(self.config, 'humans') for i in range(group_size)]
            for i, human in enumerate(group):
                p_data = self.p_data_list[p_set[i]]
                t_list = list(p_data.keys())
                last_t = max([int(t_list[i]) for i in range(len(t_list))])
                sp = [(p_data[str(t_in_real)][0] - 0.5) * self.panel_width / self.panel_scale,
                      (p_data[str(t_in_real)][1] - 0.5) * self.panel_height / self.panel_scale]
                gp = [(p_data[str(last_t)][0] - 0.5) * self.panel_width / self.panel_scale,
                      (p_data[str(last_t)][1] - 0.5) * self.panel_height / self.panel_scale]
                human.set(sp[0], sp[1], gp[0], gp[1], 0, 0, 0)
                self.human_starts.append(sp)
                self.human_goals.append(gp)
                #load real trajectory to the human policy
                if self.config.humans.policy == 'realsim_GrandCentral':
                    tra = self.p_data_list[p_set[i]]
                    trajectory = {}
                    for t, p in tra.items():
                        trajectory[int(t)] = [(p[0] - 0.5) * self.panel_width / self.panel_scale,
                                              (p[1] - 0.5) * self.panel_height / self.panel_scale]
                    human.id = p_set[i]
                    human.policy.load_trajectory(trajectory)

        else:
            raise NotImplementedError

        return group

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}
        #self.robot.set(-self.circle_radius , 0, self.circle_radius-2, -2, 0, 0, np.pi / 2)
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:

            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])


            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                if not self.current_scenario.startswith('group'):
                    human_num = self.human_num
                else:
                    group_num = self.group_num
            self.humans = []

            if not self.current_scenario.startswith('group') and not self.current_scenario.startswith('realsim'):
                for _ in range(human_num):
                    self.humans.append(self.generate_human())
            elif self.current_scenario.startswith('group'):
                self.entries = []
                self.leaves = []
                self.count_el_pair = 0
                for _ in range(group_num):
                    group_size = self.group_size
                    self.humans.extend(self.generate_group(phase, group_size))
                self.human_num = len(self.humans)
            elif self.current_scenario.startswith('realsim'):
                if self.current_scenario == 'realsim_GrandCentral':
                    # reset the intermediate variables for visualization
                    self.dynamic_human_num = []
                    self.human_goals = []
                    self.human_starts = []

                    base_seed = {'train': self.case_size['val'] + self.case_size['test'],
                                 'val': 0, 'test': self.case_size['val']}

                    GC_IMAGE_WIDTH = 1920
                    GC_IMAGE_HEIGHT = 1080
                    self.panel_height = GC_IMAGE_HEIGHT
                    self.panel_width = GC_IMAGE_WIDTH
                    self.panel_scale = 50
                    self.robot.set(0, -self.panel_height / (2 * self.panel_scale), 0, 7
                                   , 0, 0, np.pi / 2)
                    if self.p_data_list == None:
                        # read/preprocess some initialization data about grand central data sim

                        data_file = '/cs/vml4/shah/CrowdNavExt/crowd_nav/data/sim_data/GC_meta_data.json'
                        with open(data_file, 'r') as fin:
                            data = json.load(fin)
                        self.p_data_list = data['pedestrian_data_list']
                        self.f_data_list = data['frame_data_list']
                        '''
                        do some data processing
                        '''
                        count_abnormal = []
                        h_id = 1
                        for human in self.p_data_list[1:]:
                            tlist = [int(t) for t in list(human.keys())]
                            if not check_continue(tlist):
                                count_abnormal.append(h_id)
                            h_id += 1

                        '''
                        if not continue, manually add some data to it
                        '''
                        for h_id in count_abnormal:
                            ori_h = self.p_data_list[h_id]
                            after_add_positions = add_positions(ori_h)
                            self.p_data_list[h_id] = after_add_positions
                        '''
                        after changing the p_data_list, also change f_data_list
                        '''
                        new_f_data_list = make_new_f(self.p_data_list)
                        self.f_data_list = new_f_data_list[:5000]

                    t_start = 3 * (base_seed[phase] + self.case_counter[phase])
                    logging.info('current phase is: {}, t_start is: {}, self.case_counter is :{}'.format(phase, t_start, self.case_counter[phase]))
                    self.t_start = t_start
                    self.humans.extend(self.generate_group(phase, t_in_real=self.t_start))
                    self.human_num = len(self.humans)

                else:
                    raise NotImplementedError


            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        #sha:
        #potential issue of set agent'time_step and agent.policy.time_step here
        #is that when agent is not add at the method self.reset(), then...
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def nsteps_lookahead(self, actions):
        for i in range(len(actions)):
            if i <= len(actions) - 2:
                self.step(actions[i], update=True)
            else:
                return self.step(actions[i], update=False)


    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        #for realsim_*:
        first remove human from self.humans who has reached the destination,
        then add newly appeared human,
        then detect collisions,
        update environment and return
        """
        t_in_real = int(self.global_time / self.time_step) + self.t_start + 1
        if self.current_scenario.startswith('realsim'):
            # first remove human
            current_num = len(self.humans)
            remove_pid_set = []
            for i in range(current_num):
                human = self.humans[i]
                last_t = max(list(human.policy.trajectory.keys()))
                if t_in_real == last_t + 1:
                    remove_pid_set.append(i)
            remove_p_set = []
            for remove_pid in remove_pid_set:
                remove_p_set.append(self.humans[remove_pid])
            for remove_p in remove_p_set:
                self.humans.remove(remove_p)
            # then add human
            new_p_set = list(set(self.f_data_list[t_in_real]) - set(self.f_data_list[t_in_real -1]))
            if len(new_p_set) > 0:
                new_humans = self.generate_group(t_in_real=t_in_real, p_set=new_p_set)
                for human in new_humans:
                    human.time_step = self.time_step
                    human.policy.time_step = self.time_step
                self.humans.extend(new_humans)

            if update:
                self.dynamic_human_num.append(len(self.humans))
        logging.debug('at time :{}, there are :{}'.format(t_in_real, len(self.humans)))
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob, t_in_real))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                if update:
                    logging.info("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self.generate_human(human)

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
            if self.current_scenario.startswith('realsim'):
            # remove the added new human and add the removed human cause this is for lookahead
                for remove_p in remove_p_set:
                    self.humans.append(remove_p)
                if len(new_p_set)>0:
                    for human in new_humans:
                        self.humans.remove(human)

        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            # ob = []
            # for human in self.humans:
            #     if norm((self.robot.px - human.px, self.robot.py - human.py)) < self.robot_sensor_range:
            #         ob.append(human.get_observable_state())
            #
            # # if no human in the sensor range, choose the closest one
            # if not ob:
            #     distances = [norm((self.robot.px - human.px, self.robot.py - human.py)) for human in self.humans]
            #     closest_human_index = np.argmin(distances)
            #     ob.append(self.humans[closest_human_index].get_observable_state())
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())

            # # only select closest N humans
            # distances = np.array([norm((self.robot.px - human.px, self.robot.py - human.py)) for human in self.humans])
            # closest_indices = distances.argsort()[:10]
            # for index in closest_indices:
            #     ob.append(self.humans[index].get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()

        elif mode == 'scene':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_start)

            robot_goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_goal)

            if output_file is not None:
                plt.savefig(output_file)
                plt.close()

            else:
                plt.show()

        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()

        # dynamic_video
        elif mode == 'dynamic_video':
            # where there are humans add and remove during the robot navigation period, only for 'realsim_*'
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            plot_scale = 1.2
            ax.set_xlim(-self.panel_width * plot_scale / (2 * self.panel_scale), self.panel_width * plot_scale / (2 * self.panel_scale))
            ax.set_ylim(-self.panel_height * plot_scale / (2 * self.panel_scale), self.panel_height * plot_scale / (2 * self.panel_scale))
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            plot_human_radius = 0.3
            plot_robot_radius = 0.3
            human_positions = [[state[1][j].position for j in range(len(state[1]))] for state in self.states]
            human_ids = [[state[2][j] for j in range(len(state[2]))] for state in self.states]

            human_colors = [cmap(i)for i in range(len(self.human_starts))]

            for h in range(len(self.human_starts)):
                human_start = mlines.Line2D([self.human_starts[h][0]], [self.human_starts[h][1]], color='b', marker='o', linestyle='None', markersize=4)
                human_goal = mlines.Line2D([self.human_goals[h][0]], [self.human_goals[h][1]], color='r', marker='*', linestyle='None', markersize=4)

                ax.add_artist(human_start)
                ax.add_artist(human_goal)

            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], plot_robot_radius, fill=False, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            #plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers
            # Sha: here humans refer to the humans in the first frame
            humans = [plt.Circle(human_positions[0][i], plot_human_radius, fill=False, color='r')
                      for i in range(self.dynamic_human_num[0])]
            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(human_ids[0][i]),
                                          color='black') for i in range(self.dynamic_human_num[0])]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.1, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            # add human counter
            count_human = plt.text(0.6, 0.9, 'Count Human: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(count_human)

            # add reward displayer
            reward_displayer = plt.text(0.9, 1, 'r(s,a) is {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(reward_displayer)

            #add robot velocity displayer
            robot_velocity = plt.text(0.1, 1, 'v:{}, vx:{}, vy:{}'.format(0, 0, 0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(robot_velocity)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = plot_robot_radius
            orientations = []
            # sha: deal with the dynamic human number issue:
            time_step = 0
            for state in self.states:
                orientation = []
                for i in range(self.dynamic_human_num[time_step] + 1):
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                            agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                    if time_step == 0:
                        if i == 0:
                            arrow_color = 'black'
                            arrows = [
                                patches.FancyArrowPatch(*orientation[i], color=arrow_color, arrowstyle=arrow_style)]
                        else:
                            arrows.extend([patches.FancyArrowPatch(*orientation[i])])
                time_step += 1
                orientations.append(orientation)

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0


            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal humans
                nonlocal human_numbers
                nonlocal plot_human_radius

                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for human in humans:
                    human.remove()

                for human_number in human_numbers:
                    human_number.set_visible(False)
                    #human_number.remove()

                humans = [plt.Circle(human_positions[frame_num][i], plot_human_radius, fill=False, color='r')
                          for i in range(self.dynamic_human_num[frame_num])]
                if display_numbers:
                    human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(human_ids[frame_num][i]),
                                              color='black') for i in range(self.dynamic_human_num[frame_num])]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    if display_numbers:
                        ax.add_artist(human_numbers[i])

                for arrow in arrows:
                    arrow.remove()
                orientation = orientations[frame_num]
                for i in range(len(orientation)):
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[i], color='black',
                                  arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[i], color='r',
                                       arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                count_human.set_text('Count Human: {:.2f}'.format(self.dynamic_human_num[frame_num]))

                robot_action = self.robot_actions[frame_num]
                v = math.sqrt(math.pow(robot_action.vx, 2) + math.pow(robot_action.vy, 2))
                vx = robot_action.vx
                vy = robot_action.vy
                robot_velocity.set_text('v:{:.3f}, vx:{:.3f}, vy:{:.3f}'.format(v, vx, vy))

                reward_displayer.set_text('ris {}'.format(self.rewards[frame_num]))

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i-1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                with np.printoptions(precision=3, suppress=True):
                    print('A is: ')
                    print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'get_matrix_A'):
                        print_matrix_A()
                    if hasattr(self.robot.policy, 'get_feat'):
                        print_feat()
                    if hasattr(self.robot.policy, 'get_X'):
                        print_X()
                    #if hasattr(self.robot.policy, 'action_values'):
                    #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()

        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=8)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=8)
                ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i))
                      for i in range(len(self.humans))]

            # disable showing human numbers
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]

            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                        (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                           agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)
                if i == 0:
                    arrow_color = 'black'
                    arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    if display_numbers:
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i - 1) + ' '.join(
                        ['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                with np.printoptions(precision=3, suppress=True):
                    print('A is: ')
                    print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'get_matrix_A'):
                        print_matrix_A()
                    if hasattr(self.robot.policy, 'get_feat'):
                        print_feat()
                    if hasattr(self.robot.policy, 'get_X'):
                        print_X()
                    # if hasattr(self.robot.policy, 'action_values'):
                    #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()


        else:
            raise NotImplementedError
