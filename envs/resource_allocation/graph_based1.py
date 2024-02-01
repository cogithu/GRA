import numpy as np
from envs.resource_allocation.core import World, Agent
from envs.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        #world.damping = 1
        num_good_agents = 9
        num_adversaries = 0
        num_agents = num_good_agents + num_adversaries
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        self.reset_world(world)
        return world



    def reset_world(self, world):

        for agent in world.agents:
            agent.state.quantity = 1.0 + np.random.uniform(-0.01, +0.01, 1)





    def reward(self, agent, world):#单个智能体reward
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):#每个step
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0.0
        # for i, agent in enumerate(world.agents):
        if '0' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[6].state.quantity, 0.0) * min(world.agents[6].state.quantity, 0.0)
        elif '1' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[7].state.quantity, 0.0) * min(world.agents[7].state.quantity, 0.0)
        elif '5' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[8].state.quantity, 0.0) * min(world.agents[8].state.quantity, 0.0)
        elif '2' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[3].state.quantity, 0.0) * min(world.agents[3].state.quantity, 0.0)
        elif '3' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[4].state.quantity, 0.0) * min(world.agents[4].state.quantity, 0.0)
        elif '4' in agent.name:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[2].state.quantity, 0.0) * min(world.agents[2].state.quantity, 0.0)
        else:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0)

        if type(rew) is np.ndarray:
            rew = float(rew[0])
        return rew

    def observation(self, agent, world):

        down = ['3', '4', '6', '7', '8']
        if any(e in agent.name for e in down):
            j = -1.0 * np.sin(world.t)
        else:
            j = 1.0 * np.sin(world.t)
        if '0' in agent.name:
            return np.concatenate([agent.state.quantity] + [world.agents[6].state.quantity] + [[j]])#
        if '1' in agent.name:
            return np.concatenate([agent.state.quantity] + [world.agents[7].state.quantity] + [[j]])#
        if '5' in agent.name:
            return np.concatenate([agent.state.quantity] + [world.agents[8].state.quantity] + [[j]])#
        return np.concatenate([agent.state.quantity] + [agent.state.quantity] + [[j]])
