import numpy as np
from gym import spaces

class EnvCore(object):
    """
    # Agents in the environment
    """

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.world = world
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.num_agents = len(world.agents)
        if self.num_agents == 9:
            self.obs_dim = []
            for i in range(self.num_agents):
                self.obs_dim.append(3)
            self.action_dim = [2, 1, 1, 1, 1, 3, 1, 1, 1]
        # if self.agent_num == 20:
        else:
            self.obs_dim = []
            self.action_dim = []
            for i in range(self.num_agents):
                if i % 2 == 0:
                    self.obs_dim.append(4)
                    self.action_dim.append(2)
                else:
                    self.obs_dim.append(4)
                    self.action_dim.append(1)

        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agents):
            # physical action space
            u_action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.action_dim[agent],),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(u_action_space)

            # observation space
            share_obs_dim += self.obs_dim[agent]
            self.observation_space.append(
                spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.obs_dim[agent],),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agents)
        ]


    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self.world.t = 0
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}


        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            agent.action.output = actions[i]

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case

        return [obs_n, reward_n, done_n, info_n]
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
