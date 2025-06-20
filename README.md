# GRA

### How to Run

The `train.py` script serves as a key component for training multiple agents across various scenarios using different algorithms, including `gra` and `mappo`. You can configure the script to utilize either GPU or CPU for the training process.

#### Running the Script
You can initiate the `train.py` script using the following command:

```bash
python GRA/train/train.py [arguments]
```

#### Important Arguments
Here are some crucial arguments that you might need to set:

- **`--algorithm_name`**: Designate the algorithm to be employed. Available options include `rmappo` and `mappo`.
- **`--environment_name`**: Specify the training environment. Supported environments are `resource_allocation`, `mpe`, and `ReplenishmentEnv`.
- **`--scenario_name`**: Select a specific scenario within the chosen environment, such as `graph_based1`, `encircle`, etc.

#### Example Command
For instance, if you wish to use the `gra` algorithm to train agents in the `resource_allocation` environment with the `graph_based1` scenario, you can execute the following command:

```bash
python GRA/train/train.py --algorithm_name gra --environment_name resource_allocation --scenario_name graph_based1
```
