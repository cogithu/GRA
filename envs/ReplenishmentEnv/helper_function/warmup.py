import numpy as np

"""
    Warm up the env before action in look back length.
    In warm up process, replenish quantity will be set to last day's demand.
"""
def replenish_by_last_demand(env) -> None:
    _init_balance = env.balance.copy()
    _init_per_balance = env.per_balance.copy()
    _action_mode = env.action_mode

    env.action_mode = "continuous" # demand_mean_discrete
    # last_demand = np.zeros((len(env.warehouse_list), len(env.sku_list)))
    sum_last_demand = []
    for i in range(len(env.warehouse_list)):
        init_demand = np.zeros(len(env.sku_list) * env.act_dim[i])
        sum_last_demand.append(init_demand)

    for day in range(env.lookback_len):
        env.step(sum_last_demand)
        last_demand = env.agent_states[-1, "all_warehouses", "demand", "yesterday"]
        sum_last_demand = []
        for i in range(len(env.warehouse_list)):
            sum_demand = np.sum(last_demand[i], axis=-1, keepdims=True) / env.act_dim[i]
            sum_last_demand.append(np.tile(sum_demand, (1, env.act_dim[i])))

    env.action_mode = _action_mode
    env.balance = _init_balance.copy()
    env.per_balance = _init_per_balance.copy()
