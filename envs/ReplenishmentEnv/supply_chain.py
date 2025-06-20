"""
    SupplyChain class, contains a list of warehouses.
    Currently, only chainlike supply chain are supported. 
    Each warehouse has only 1 upstream and 1 downstream.
    Head warehouse receive skus from super_vendor and tail warehouse sells skus to consumer.
"""
class SupplyChain:
    def __init__(self, warehouse_config) -> None:
        """
            self.warehouse_dict = {
                warehouse_name: [upstream_name, downstream_name, capacity],
                warehouse_name: [upstream_name, downstream_name, capacity],
                ...
            }
            Warehouse's index represents the order as actions, rewards, and agent_states in env.
        """
        self.warehouse_dict = {}
        self.head = []
        self.tail = []
        self.consumer = "consumer"
        self.super_vendor = "super_vendor"

        for warehouse in warehouse_config:
            warehouse_name = warehouse["name"]
            # 修改 upstream 和 downstream 为列表
            upstreams = warehouse.get("upstream", [])
            downstreams = warehouse.get("downstream", [])

            # 记录头节点和尾节点
            if self.super_vendor in upstreams:
                self.head.append(warehouse_name)
            if self.consumer in downstreams:
                self.tail.append(warehouse_name)

            self.warehouse_dict[warehouse_name] = {
                "upstream": upstreams,
                "downstream": downstreams,
                "capacity": warehouse.get("capacity", 0),
                "init_balance": warehouse.get("init_balance", 0),
                "unit_storage_cost": warehouse.get("unit_storage_cost", 0),
                "accept_sku": warehouse.get("accept_sku", "equal_accept")
            }

            # 检查供应链的有效性（防止循环）
        # self._validate_supply_chain()

    def _validate_supply_chain(self):
        visited = set()
        for warehouse in self.warehouse_dict:
            if warehouse not in visited:
                self._check_cycles(warehouse, visited, [])

    def _check_cycles(self, current, visited, path):
        if current in path:
            raise ValueError(f"Cycle detected in supply chain: {path}")
        if current in visited:
            return
        visited.add(current)
        path.append(current)
        for downstream in self.warehouse_dict[current]["downstream"]:
            if downstream != self.consumer:
                self._check_cycles(downstream, visited, path.copy())
        path.pop()
        # Check for supply chain
        assert(self.head is not None)
        assert(self.tail is not None)
        for warehouse, value in self.warehouse_dict.items():
            assert(isinstance(value["upstream"], str))
            assert(isinstance(value["upstream"], str))
            if self.warehouse_dict[warehouse]["upstream"] != "super_vendor":
                assert(self.warehouse_dict[self.warehouse_dict[warehouse]["upstream"]]["downstream"] == warehouse)
            if self.warehouse_dict[warehouse]["downstream"] != "consumer":
                assert(self.warehouse_dict[self.warehouse_dict[warehouse]["downstream"]]["upstream"] == warehouse)

    # index = [warehouse_name, item].
    # item includes upstream, downstream, capacity, init_balance and unit_storage_cost
    def __getitem__(self, index):
        assert(isinstance(index, tuple))
        assert(len(index) == 2)
        warehouse_name = index[0]
        item = index[1]
        assert(warehouse_name in self.warehouse_dict)
        assert(item in self.warehouse_dict[warehouse_name])
        return self.warehouse_dict[warehouse_name][item]
    
    # Return the head index, which upstream is super_vendor
    def get_head(self) -> int:
        return self.head

    # Return the tail index, which downstream is consumer
    def get_tail(self) -> int:
        return self.tail

    def get_warehouse_count(self) -> int:
        return len(self.warehouse_dict)

    def get_warehouse_list(self) -> list:
        return list(self.warehouse_dict.keys())
    
    def get_consumer(self) -> str:
        return self.consumer
    
    def get_super_vendor(self) -> str:
        return self.super_vendor
