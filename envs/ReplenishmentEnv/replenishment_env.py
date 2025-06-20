import copy
import os
from datetime import datetime, timedelta
from gym import Env, spaces
from typing import Tuple
import numpy as np
import pandas as pd
import random
import pandas as pd
import yaml


from envs.ReplenishmentEnv.helper_function.rewards import reward1, reward2
from envs.ReplenishmentEnv.helper_function.warmup import replenish_by_last_demand
from envs.ReplenishmentEnv.helper_function.convertors import continuous, discrete, demand_mean_continuous, demand_mean_discrete
from envs.ReplenishmentEnv.helper_function.accept import equal_accept
from envs.ReplenishmentEnv.supply_chain import SupplyChain
from envs.ReplenishmentEnv.agent_states import AgentStates
from envs.ReplenishmentEnv.utility.utility import deep_update
from envs.ReplenishmentEnv.utility.visualizer import Visualizer
from envs.ReplenishmentEnv.utility.data_loader import DataLoader


class ReplenishmentEnv(Env):
    def __init__(self, config_path, mode="train", vis_path=None, update_config=None):
        # Config loaded from yaml file
        self.config: dict = None

        # All following variables will be refresh in init_env function
        # Sku list from config file
        self.sku_list: list = None

        # Duration in days
        self.durations = 0

        # Look back information length
        self.lookback_len = 0

        # All warehouses' sku data are shored in total_data, including 3 types of sku information.
        # self.total_data = [
        # {
        #       "warehouse_name" : name,
        #       "shared_data"   : shared_data, 
        #       "static_data"   : static_data, 
        #       "dynamic_data"  : dynamic_data
        # },
        #    ...
        #]
        # Shared info saved shared information for all skus and all dates.
        # Stored as dict = {state item: value} 
        # Static info saved special information for each sku but will not changed.
        # Stored as N * M pd.DataFrame (N: agents_count, M: state item count)
        # Dynamic info saved the information which is different between different skus and will change by date.
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.total_data: list = None

        # Step tick.
        self.current_step = 0

        # Env mode including train, validation and test. Each mode has its own dataset.
        self.mode = mode

        # visualization path
        self.vis_path = os.path.join("output", datetime.now().strftime('%Y%m%d_%H%M%S')) if vis_path is None else vis_path
 
        self.load_config(config_path, update_config)
        self.build_supply_chain()

        self.load_data()
        self.act_dim = []
        self.sale_dim = []
        for warehouse in self.supply_chain.get_warehouse_list():
            self.act_dim.append(len(self.supply_chain[warehouse, "upstream"]))
            self.sale_dim.append(len(self.supply_chain[warehouse, "downstream"]))
        self.init_env()
        self.warm_flag = 0
        self.flag = [0 for _ in range(len(self.act_dim))]

    def load_config(self, config_path: str, update_config: dict) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
    
        if update_config is not None:
            self.config = deep_update(self.config, update_config)

        assert("env" in self.config)
        assert("mode" in self.config["env"])
        assert("sku_list" in self.config["env"])
        assert("warehouse" in self.config)
        assert("profit_function" in self.config)
        assert("reward_function" in self.config)
        assert("output_state" in self.config)
        assert("action" in self.config)
        assert("visualization" in self.config)

    """
        Build supply chain
    """
    def build_supply_chain(self) -> None:
        self.supply_chain   = SupplyChain(self.config["warehouse"])
        self.warehouse_list  = self.supply_chain.get_warehouse_list()
        self.warehouse_to_id = {warehouse_name: index for index, warehouse_name in enumerate(self.warehouse_list)}
        self.id_to_warehouse = {index: warehouse_name for index, warehouse_name in enumerate(self.warehouse_list)}
    
    """
        Load all warehouses' sku data, including warehouse name, shared data, static data and dynamic data.
        Only load only in __init__ function.
    """
    def load_data(self) -> None:
        self.total_data = []
        data_loader = DataLoader()
        # Convert the sku list from file to list. 
        if isinstance(self.config["env"]["sku_list"], str):
            self.sku_list = data_loader.load_as_list(self.config["env"]["sku_list"])
        else:
            self.sku_list = self.config["env"]["sku_list"]

        for warehouse_config in self.config["warehouse"]:
            assert("sku" in warehouse_config)
            warehouse_name   = warehouse_config["name"]
            sku_config      = warehouse_config["sku"]

            # Load shared info, which is shared for all skus and all dates.
            # Shared info is stored as dict = {state item: value}
            warehouse_shared_data = sku_config.get("shared_data", {})

            # Load and check static sku info, which is special for each sku but will not changed.
            # Static info is stored as N * M pd.DataFrame (N: agents_count, M: state item count)
            if "static_data" in sku_config:
                warehouse_static_data = data_loader.load_as_df(sku_config["static_data"])
            else:
                warehouse_static_data = pd.DataFrame(columns=["SKU"])
                warehouse_static_data["SKU"] = self.sku_list

            # Load and check demands info, which is different between different skus and will change by date
            # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
            warehouse_dynamic_data = {}
            for dynamic_data_item in sku_config.get("dynamic_data", {}):
                assert("name" in dynamic_data_item)
                assert("file" in dynamic_data_item)
                item = dynamic_data_item["name"]
                file = dynamic_data_item["file"]
                dynamic_value = data_loader.load_as_matrix(file)
                warehouse_dynamic_data[item] = dynamic_value

            self.total_data.append({
                "warehouse_name" : warehouse_name,
                "shared_data"   : warehouse_shared_data, 
                "static_data"   : warehouse_static_data, 
                "dynamic_data"  : warehouse_dynamic_data
            })
    
    """
        Init and transform the shared, static and dynamic data by:
        - Remove useless sku in all sku data
        - Remove useless date in all sku data
        - Rename the date to step 
        - Check sku and date in static and dynamic data
        init_data will be called in reset function.
    """
    def init_data(self) -> None:
        self.picked_data = []
        for data in self.total_data:
            picked_warehouse_data = {"warehouse_name": data["warehouse_name"]}

            # Load shared info
            picked_warehouse_data["shared_data"] = data["shared_data"]

            # Load static info
            picked_warehouse_data["static_data"] = data["static_data"]
            if "SKU" in data["static_data"]:
                assert(set(self.sku_list) <= set(list(data["static_data"]["SKU"].unique())))
                # Remove useless sku
                picked_warehouse_data["static_data"] = data["static_data"][data["static_data"]["SKU"].isin(self.sku_list)]

            # Load and check demands info, which is different between different skus and will change by date
            # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
            picked_warehouse_data["dynamic_data"] = {}
            for item, ori_dynamic_value in data["dynamic_data"].items():
                dynamic_value = copy.deepcopy(ori_dynamic_value)
                assert(set(self.sku_list) <= set(dynamic_value.columns.unique()))
                dates_format_dic = {
                    datetime.strftime(pd.to_datetime(date), "%Y/%m/%d"): date
                    for date in dynamic_value.index
                }
                self.date_to_index = {}
                # For warmup, start date forwards lookback_len
                for step, date in enumerate(pd.date_range(self.picked_start_date - timedelta(self.lookback_len), self.picked_end_date)):
                    date_str = datetime.strftime(date, "%Y/%m/%d")
                    # Check the target date in saved in demands file
                    assert(date_str in dates_format_dic.keys())
                    # Convert the actual date to date index
                    self.date_to_index[dates_format_dic[date_str]] = step - self.lookback_len
                dynamic_value.rename(index=self.date_to_index, inplace=True)
                # Remove useless date and sku
                dynamic_value = dynamic_value.loc[self.date_to_index.values()]
                dynamic_value = dynamic_value[self.sku_list]
                picked_warehouse_data["dynamic_data"][item] = dynamic_value.sort_values(by="Date")
            self.picked_data.append(picked_warehouse_data)

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update
    """
    def reset(self) -> None:
        self.init_env()
        self.init_data()
        self.init_state()
        self.init_monitor()
        self.warm_flag = 0
        eval(format(self.warmup_function))(self)
        states = self.get_state()
        states = states.reshape(self.warehouse_count, self.sku_count * (len(self.current_output_state) + len(self.lookback_output_state) * self.lookback_len))
        self.warm_flag = 1
        return states  
    
    def init_env(self) -> None:
        # Get basic env info from config
        self.sku_count              = len(self.sku_list)
        self.warehouse_count        = self.supply_chain.get_warehouse_count()
        self.agent_count            = self.sku_count * self.warehouse_count
        self.integerization_sku     = self.config["env"].get("integerization_sku", False)
        self.lookback_len           = self.config["env"].get("lookback_len", 7)
        self.current_output_state   = self.config["output_state"].get("current_state", [])
        self.lookback_output_state  = self.config["output_state"].get("lookback_state", [])
        self.action_mode            = self.config["action"].get("mode", "continuous")
        self.warmup_function        = self.config["env"].get("warmup", "replenish_by_last_demand")
        self.balance                = [self.supply_chain[warehouse, "init_balance"] for warehouse in self.warehouse_list]
        self.per_balance            = np.zeros(self.agent_count)

        # Get mode related info from mode config
        mode_configs = [mode_config for mode_config in self.config["env"]["mode"] if mode_config["name"] == self.mode]
        assert(len(mode_configs) == 1)
        self.mode_config = mode_configs[0]
        assert("start_date" in self.mode_config)
        assert("end_date" in self.mode_config)
        self.start_date             = pd.to_datetime(self.mode_config["start_date"])
        self.end_date               = pd.to_datetime(self.mode_config["end_date"])
        self.random_interception    = self.mode_config.get("random_interception", False)

        # Step tick. Due to warmup, step starts from -self.lookback_len.
        self.current_step = -self.lookback_len
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        output_length = len(self.current_output_state) + len(self.lookback_output_state) * self.lookback_len
        upstream_num = []
        for i in range(self.warehouse_count):
            self.action_space.append(spaces.Box(
                low=0, high=np.inf,
                shape=(self.sku_count * self.act_dim[i], ), dtype=np.float32
            ))
            # self.action_space.append(spaces.MultiDiscrete([34] * self.act_dim[i]))
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.sku_list) * output_length, ),
                dtype=np.float32
            ))
            self.share_observation_space.append(spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.sku_list) * output_length * self.warehouse_count, ),
                dtype=np.float32
            ))
        # Current all agent_states will be returned as obs.

        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(output_length, len(self.sku_list)),
        #     dtype=np.float32
        # )

        if self.random_interception:
            self.picked_start_date, self.picked_end_date = self.interception()
        else:
            self.picked_start_date, self.picked_end_date = self.start_date, self.end_date
        self.durations = (self.picked_end_date - self.picked_start_date).days + 1

    def interception(self) -> Tuple[datetime, datetime]:
        horizon = self.config["env"].get("horizon", 100)
        date_length = (self.end_date - self.start_date).days + 1
        start_date_index = random.randint(0, date_length - horizon)
        picked_start_date = self.start_date + timedelta(start_date_index)
        picked_end_date = picked_start_date + timedelta(horizon - 1)
        return picked_start_date, picked_end_date

    def init_state(self) -> None:  # 把picked_data填入state
        self.agent_states = AgentStates(
            self.act_dim,
            self.sale_dim,
            self.sku_list, 
            self.durations,
            self.picked_data,
            self.lookback_len,
        )

    def init_monitor(self) -> None:
        state_items = self.config["visualization"].get("state", ["replenish", "demand"])
        self.reward_info_list = []
        self.visualizer = Visualizer(
            self.agent_states, 
            self.reward_info_list,
            self.picked_start_date, 
            self.sku_list, 
            self.warehouse_list,
            state_items,
            self.vis_path,
            self.lookback_len
        )
        


    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: C * N matrix, C warehouse count, N agent count
        contains action_idx or action_quantity, defined by action_setting in config
    """
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        self.flag = [0 for _ in range(len(self.act_dim))]
        for i in range(self.warehouse_count):
            actions[i] = actions[i].reshape(self.sku_count, -1)
        self.replenish(actions)
        self.sell()
        self.receive_sku()
        self.profit, reward_info = self.get_reward()
        self.balance += np.sum(self.profit, axis=1)
        self.per_balance += self.profit.flatten()

        states = self.get_state()
        if self.warm_flag == 1:
            states = states.reshape(self.warehouse_count, self.sku_count * (
                        len(self.current_output_state) + len(self.lookback_output_state) * self.lookback_len))
        rewards, reward_info = self.get_reward()
        self.reward_info_list.append(reward_info)
        infos = self.get_info()
        infos["reward_info"] = reward_info

        self.next_step()
        done = []
        dones = self.current_step >= self.durations
        for _ in range(self.warehouse_count):
            done.append(dones)

        return states, rewards, done, infos

    """
        Sell by demand and in stock
    """
    def sell(self) -> None:
        for warehouse in self.supply_chain.get_warehouse_list():
            current_demand = self.agent_states[self.warehouse_to_id[warehouse], warehouse, "demand"]  # (sku_num,n)
            in_stock = self.agent_states[warehouse, "in_stock"]
            demand_sum = np.sum(current_demand, axis=-1)
            sale = np.zeros_like(current_demand)
            for i in range(current_demand.shape[0]):
                if demand_sum[i] < in_stock[i]:
                    sale[i] = current_demand[i]
                else:
                    weights = self.softmax(current_demand[i])
                    sale[i] = weights * in_stock[i]
            self.agent_states[self.warehouse_to_id[warehouse], warehouse, "sale"] = sale
            self.agent_states[warehouse, "in_stock"] -= np.sum(sale, axis=-1)

        # Sold sku will arrive downstream warehouse after vlt dates 
        for warehouse in self.supply_chain.get_warehouse_list():
            downstreams = self.supply_chain[warehouse, "downstream"]
            for i, downstream in enumerate(downstreams):
                if downstream == self.supply_chain.consumer:
                    continue
                vlt = self.agent_states[downstream, "vlt"]

                max_future = min(int(max(vlt)) + 1, self.durations - self.agent_states.current_step)
                arrived_index = np.array(range(0, max_future, 1)).reshape(-1, 1)
                arrived_flag = np.where(arrived_index == vlt, 1, 0)

                arrived_matrix = arrived_flag * self.agent_states[self.warehouse_to_id[warehouse], warehouse, "sale"][:, i]
                future_dates = np.array(range(0, max_future, 1)) + self.agent_states.current_step
                self.agent_states[downstream, "arrived", future_dates] += arrived_matrix
                self.agent_states[downstream, "in_transit"] += self.agent_states[self.warehouse_to_id[warehouse], warehouse, "sale"][:, i]


    """
        Receive the arrived sku.
        When exceed the storage capacity, sku will be accepted in same ratio
    """
    def receive_sku(self) -> None:
        for warehouse in self.supply_chain.get_warehouse_list():
            # Arrived from upstream
            arrived = self.agent_states[warehouse, "arrived"]
            self.agent_states[warehouse, "in_transit"] -= arrived
            capacity = self.supply_chain[warehouse, "capacity"]
            accept_amount = eval(self.supply_chain[warehouse, "accept_sku"])(arrived, capacity, self.agent_states, warehouse)
            if self.integerization_sku:
                accept_amount = np.floor(accept_amount)
            
            self.agent_states[warehouse, "excess"] = arrived - accept_amount
            self.agent_states[warehouse, "accepted"] = accept_amount
            self.agent_states[warehouse, "in_stock"] += accept_amount

    """
        Replenish skus by actions
        actions: [action_idx/action_quantity] by sku order, defined by action setting in config
    """
    def replenish(self, actions) -> None:
        replenish_amount = eval(self.action_mode)(actions, self.config["action"], self.agent_states)

        if self.integerization_sku:
            for i, value in enumerate(replenish_amount):
                replenish_amount[i] = np.floor(value)
        
        # Warehouse's replenishment is the replenishment as upstream's demand.
        for warehouse in self.supply_chain.get_warehouse_list():
            warehouse_replenish_amount = replenish_amount[self.warehouse_to_id[warehouse]]
            self.agent_states[self.warehouse_to_id[warehouse], warehouse, "replenish"] = warehouse_replenish_amount  # all replenish
            # upstream = self.supply_chain[warehouse, "upstream"]
            upstreams = self.supply_chain[warehouse, "upstream"]
            for i, upstream in enumerate(upstreams):  # todo
                if upstream == self.supply_chain.super_vendor:  # super_vendor
                    # If upstream is super_vendor, all replenishment will arrive after vlt dates
                    vlt = self.agent_states[warehouse, "vlt"]
                    max_future = min(int(max(vlt)) + 1, self.durations - self.agent_states.current_step)
                    arrived_index = np.array(range(0, max_future, 1)).reshape(-1, 1)
                    arrived_flag = np.where(arrived_index == vlt, 1, 0)
                    arrived_matrix = arrived_flag * self.agent_states[self.warehouse_to_id[warehouse], warehouse, "replenish"][:, i]
                    future_dates = np.array(range(0, max_future, 1)) + self.agent_states.current_step
                    self.agent_states[warehouse, "arrived", future_dates] += arrived_matrix
                    self.agent_states[warehouse, "in_transit"] += warehouse_replenish_amount[:, i]
                else:
                    # If upstream is not super_vendor, all replenishment will be sent as demand to upstream
                    if self.agent_states.current_step < self.durations - 1:
                        self.agent_states.demand_list[self.warehouse_to_id[upstream]][(self.current_step + self.lookback_len), :, self.flag[self.warehouse_to_id[upstream]]] = warehouse_replenish_amount[:, i]
                        self.flag[self.warehouse_to_id[upstream]] += 1
                        self.agent_states[upstream, "demand"] += warehouse_replenish_amount[:, i]

    def get_reward(self) -> Tuple[np.array, dict]:
        reward_info = {
            "unit_storage_cost": [self.supply_chain[warehouse, "unit_storage_cost"] for warehouse in self.warehouse_list]
        }
        reward_info = eval(self.config["reward_function"])(self.agent_states, reward_info)
        rewards = reward_info["reward"]
        # rewards = np.sum(rewards, axis=1)
        return rewards, reward_info

    # Output C * N * M matrix:  C is warehouse count, N is sku count and M is state count
    def get_state(self) -> dict:
        states = self.agent_states.snapshot(self.current_output_state, self.lookback_output_state)
        return states

    def softmax(self, data):
        sum_exp_data = np.sum(data) + 1e-8
        softmax_data = data / sum_exp_data
        return softmax_data

    def get_info(self) -> dict:
        info = {
            "profit": self.profit,
            "balance": self.balance
        }
        return info
    
    def pre_step(self) -> None:
        self.current_step -= 1
        self.agent_states.pre_step()

    def next_step(self) -> None:
        self.current_step += 1
        self.agent_states.next_step()

    def render(self) -> None:
        self.visualizer.render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

