import numpy as np
from envs.resource_allocation.core import World, Agent
from envs.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = 20
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.num = i

        self.reset_world(world)
        return world


    def reset_world(self, world):

        for agent in world.agents:
            agent.state.quantity = 1 + np.random.uniform(-0.01, +0.01, 1)


    def reward(self, agent, world):#单个智能体reward
        # Agents are rewarded based on minimum agent distance to each landmark
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):#每个step
        # Agents are rewarded based on minimum agent distance to each landmark
        # rew = 0
        # for i, agent in enumerate(world.agents):
        if agent.num == 0:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[1].state.quantity, 0.0) * min(world.agents[1].state.quantity, 0.0)\
                   - min(world.agents[len(world.agents) - 1].state.quantity, 0.0) * min(world.agents[len(world.agents) - 1].state.quantity, 0.0)
        elif agent.num % 2 == 0:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0) \
                  - min(world.agents[agent.num - 1].state.quantity, 0.0) * min(world.agents[agent.num - 1].state.quantity, 0.0) \
                   - min(world.agents[agent.num + 1].state.quantity, 0.0) * min(world.agents[agent.num + 1].state.quantity, 0.0)
        else:
            rew = - min(agent.state.quantity, 0.0) * min(agent.state.quantity, 0.0)
        if type(rew) is np.ndarray:
            rew = float(rew[0])
        return rew

        # return rew

    def observation(self, agent, world):
        j = 0.5 * np.sin(world.t)
        if agent.num == len(world.agents) - 1:
            return np.concatenate([agent.state.quantity] + [world.agents[agent.num - 1].state.quantity]
                                  + [world.agents[0].state.quantity] + [[-j]])
        if agent.num % 2 == 1:
            return np.concatenate([agent.state.quantity] + [world.agents[agent.num - 1].state.quantity]
                                  + [world.agents[agent.num + 1].state.quantity] + [[-j]])


        # initpos = []
        # for other in world.agents:
        #     initpos.append(other.state.initpos)

        return np.concatenate([agent.state.quantity] + [agent.state.quantity] + [agent.state.quantity]
                              + [[-j]])

        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)