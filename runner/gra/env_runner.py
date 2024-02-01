import time
import os
import numpy as np
from itertools import chain
import torch
import pickle
from utils.util import update_linear_schedule
from runner.gra.base_runner import Runner
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.x = []
        self.y = []
        self.save_path = self.all_args.save_dir + '/' + self.all_args.scenario_name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self.warmup()
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    actions,
                    action_log_probs,
                    rnn_states,
                    # rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs

                obs, rewards, dones, infos = self.envs.step(actions_env)

                values, rnn_states_critic = self.get_value(step, rewards)

                # time.sleep(0.1)
                # self.envs.render()

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            rewards = self.get_next_rewards(self.episode_length)

            # compute return and update network
            self.compute(rewards)

            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                        train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )

            # eval
            if episode % 10 == 0:
                self.eval(total_num_steps)


    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = obs
        # for o in obs:
        #     share_obs.append(list(chain(*o)))
        # share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        # rnn_states_critic = []
        share_obs = []

        for agent_id in range(self.num_agents):
            share_obs.append(list(self.buffer[agent_id].obs[step]))
        share_obs = np.array(share_obs).transpose(1, 0, 2)
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            action, action_log_prob, rnn_state = self.trainer[
                agent_id
            ].policy.get_actions(
                share_obs,
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                # self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            action = _t2n(action)
            action = list(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            # rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)
        if len(np.array(actions).shape) == 2:
            actions = np.array(actions).transpose(1, 0)
        else:
            actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        # rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            actions,
            action_log_probs,
            rnn_states,
            # rnn_states_critic,
            actions_env,
        )

    def get_value(self, step, rewards):
        values = []
        rnn_states_critic = []
        share_obs = []

        for agent_id in range(self.num_agents):
            share_obs.append(list(self.buffer[agent_id].obs[step]))
        share_obs = np.array(share_obs).transpose(1, 0, 2)
        rewards = rewards.reshape(self.n_rollout_threads, self.num_agents, 1)
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, rnn_state_critic = self.trainer[agent_id].policy.get_values(
                share_obs,
                rewards,
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            rnn_states_critic.append(_t2n(rnn_state_critic))
        values = np.array(values).transpose(1, 0, 2)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            rnn_states_critic,
        )

    def get_next_rewards(self, step):
        (
            actions,
            action_log_probs,
            rnn_states,
            # rnn_states_critic,
            actions_env,
        ) = self.collect(step)

        # Obser reward and next obs
        obs, rewards, dones, infos = self.envs.step(actions_env)
        return rewards

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = obs
        share_rewards = rewards


        index = []
        edge_attr = [[]for i in range(self.num_agents)]

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            for j in range(len(self.all_args.learn_index[agent_id])):
                index.append([j, agent_id])
                edge_attr[agent_id].append(rewards[:, j])

            self.buffer[agent_id].insert(
                share_obs,
                share_rewards,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                np.array(list(actions[:, agent_id])),
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []

        eval_obs = self.envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    eval_action_env = eval_action
                #     raise NotImplementedError

                eval_temp_actions_env.append(list(eval_action_env))
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            actions_env = []
            for i in range(self.n_rollout_threads):
                one_hot_action_env = []
                for temp_actions_env in eval_temp_actions_env:
                    one_hot_action_env.append(temp_actions_env[i])
                actions_env.append(one_hot_action_env)

            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(actions_env)

            # Obser reward and next obs
            # eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)



        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_episode_rewards = np.sum(eval_episode_rewards) / self.n_rollout_threads / self.episode_length

        self.y.append(eval_episode_rewards)

        save_variable(self.y, 'new92_y')
        plt.figure()
        plt.plot(range(len(self.y)), self.y)
        plt.xlabel('episode')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/pltnew92.png', format='png')
        plt.close("all")
        print("eval average episode rewards : " + str(eval_episode_rewards))



    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                # rnn_states[dones == True] = np.zeros(
                #     ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                #     dtype=np.float32,
                # )
                # masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                # masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
def save_variable(v, filename):
    f = open(filename, 'wb')  # 打开或创建名叫filename的文档。
    pickle.dump(v, f)  # 在文件filename中写入v
    f.close()  # 关闭文件，释放内存。


def load_variavle(filename):
    try:
        f = open(filename, 'rb+')
        r = pickle.load(f)
        f.close()
        return r
    except EOFError:
        return ""