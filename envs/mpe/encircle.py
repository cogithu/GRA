import numpy as np
from envs.mpe.core import World, Agent, Landmark
from envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment
import random

#
#     # the non-ensemble version of <ensemble_push>
#
#

def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None]*len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas

def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle<0:
        angle += 2*np.pi
    return angle


class Scenario(BaseScenario):
    def __init__(self):
        self.num_agents = 10
        self.target_radius = 0.7  # fixing the target radius for now
        self.target_ang = 2 * np.pi / self.num_agents
        # self.dis = self.target_radius * 2 * np.sin(np.pi/self.num_agents)
        self.ideal_theta_separation = (2 * np.pi) / self.num_agents * 2  # ideal theta difference between two agents

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 10
        num_adversaries = 0
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.num = i
            agent.collide = False
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8
            landmark.index = i
        # set goal landmark
        goal = world.landmarks[0]
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            if i == 0:
                agent.color = np.array([0.25, 0.25, 0.75])
            elif i % 2 == 0:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.color = np.array([0.25, 0.75, 0.25])
            # if agent.adversary:
            #     agent.color = np.array([0.75, 0.25, 0.25])
            # else:
            #     j = goal.index
            #     agent.color[j + 1] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark
    #     return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        if agent.num % 2 == 1:
            landmark_pose = world.landmarks[0].state.p_pos
            relative_poses = [agent.state.p_pos - landmark_pose for agent in world.agents]
            thetas = get_thetas(relative_poses)
            # anchor at the agent with min theta (closest to the horizontal line)
            theta_min = min(thetas)
            expected_poses = [landmark_pose + self.target_radius * np.array(
                [np.cos(theta_min + i * self.ideal_theta_separation),
                 np.sin(theta_min + i * self.ideal_theta_separation)])
                              for i in range(int(self.num_agents/2))]

            rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - expected_poses[int((agent.num - 1)/2)])))

        elif agent.num == 0:
            rew = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
                     - 2 * np.square(np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[1].state.p_pos))) \
                                     - np.sqrt(
                np.sum(np.square(agent.state.p_pos - world.agents[len(world.agents) - 1].state.p_pos))))
        elif agent.num % 2 == 0:
            rew = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
                     - 2 * np.square(
                np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[agent.num - 1].state.p_pos))) \
                - np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[agent.num + 1].state.p_pos))))
            # dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in expected_poses] for a in world.agents])
            # # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            # self.delta_dists = self._bipartite_min_dists(dists)
            # world.dists = self.delta_dists
            #
            # total_penalty = np.mean(np.clip(self.delta_dists, 0, 2))
            # self.joint_reward = -total_penalty

        return rew

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def _get_angle(self, agent, world):
        # compute angle
        vector1 = agent.state.p_pos - agent.goal_a.state.p_pos
        if agent.num == 0:
            vector2 = world.agents[len(world.agents) - 1].state.p_pos - agent.goal_a.state.p_pos
            vector3 = world.agents[agent.num + 1].state.p_pos - agent.goal_a.state.p_pos
        else:
            vector2 = world.agents[agent.num - 1].state.p_pos - agent.goal_a.state.p_pos
            vector3 = world.agents[agent.num + 1].state.p_pos - agent.goal_a.state.p_pos

        angle1 = np.arccos(np.dot(vector1, vector2)/(np.sqrt(np.dot(vector1, vector1))*np.sqrt(np.dot(vector2, vector2))))
        angle2 = np.arccos(np.dot(vector1, vector3)/(np.sqrt(np.dot(vector1, vector1))*np.sqrt(np.dot(vector3, vector3))))
        if np.cross(vector1, vector2) < 0:
            angle1 = - angle1
        if np.cross(vector1, vector3) < 0:
            angle2 = - angle2
        return angle1, angle2

    def reward(self, agent, world):
        # the distance to the goal
        if agent.num == 0:
            ang1, ang2 = self._get_angle(agent, world)
            reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
                     - np.square(-self.target_ang - ang1) \
                     - np.square(self.target_ang - ang2)
        elif agent.num % 2 == 0:
            ang1, ang2 = self._get_angle(agent, world)
            reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
                     - np.square(-self.target_ang - ang1) \
                     - np.square(self.target_ang - ang2)
        else:
            reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))))
        # if agent.num == 0:
        #     reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
        #              - np.square(self.dis - np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[1].state.p_pos)))) \
        #              - np.square(self.dis - np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[len(world.agents) - 1].state.p_pos))))
        # elif agent.num % 2 == 0:
        #     reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))) \
        #              - np.square(self.dis - np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[agent.num - 1].state.p_pos)))) \
        #              - np.square(self.dis - np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[agent.num + 1].state.p_pos))))
        # else:
        #     reward = -np.square(self.target_radius - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))))
        return reward

    def adversary_reward(self, agent, world):
        # keep the nearest good agents away from the goal
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        pos_rew = min(agent_dist)
        #nearest_agent = world.good_agents[np.argmin(agent_dist)]
        #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
        #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        return pos_rew - neg_rew
               
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        if agent.num == len(world.agents) - 1:
            other_pos.append(world.agents[agent.num - 1].state.p_pos - agent.state.p_pos)
            other_pos.append(world.agents[0].state.p_pos - agent.state.p_pos)
        elif agent.num % 2 == 1:
            other_pos.append(world.agents[agent.num - 1].state.p_pos - agent.state.p_pos)
            other_pos.append(world.agents[agent.num + 1].state.p_pos - agent.state.p_pos)
        else:
            other_pos.append(entity.state.p_pos - agent.state.p_pos)
            other_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)

