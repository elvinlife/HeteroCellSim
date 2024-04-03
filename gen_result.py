#!/usr/bin/python3
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def provision_metric(weight, ideal_weight):
    diff = weight - ideal_weight
    if diff > 0.001:
        return 1
    elif diff < -0.001:
        return -1
    else:
        return 0


class UEInfo:
    def __init__(self, ue_id, cqi_all, coord_x, coord_y):
        """
        self.cqi_all: [[0, datarate_c0], [1, datarate_c1], [2, datarate_c2]]
        """
        self.ue_id = ue_id
        self.cqi_all = cqi_all
        self.coord_x = coord_x
        self.coord_y = coord_y

    def __str__(self) -> str:
        ret_str = "("
        for cqi in self.cqi_all:
            ret_str = ret_str + str(cqi[1]) + ","
        ret_str = ret_str + ")"
        return ret_str

    def set_sid(self, sid):
        self.sid = sid

    def get_cqi(self, cid):
        return self.cqi_all[cid][1]


class Simulator:
    def __init__(self, n_cells: int, n_slices: int) -> None:
        """
        self.ue_info: a 2D array, self.ue_info[cid][sid] stores the k&v pair of UEs
        key is the ue_id, and value is the UEInfo
        """
        self.n_cells = n_cells
        self.n_slices = n_slices
        self.ue_info = []
        for i in range(n_cells):
            cell_ues = []
            for j in range(n_slices):
                cell_ues.append({})
            self.ue_info.append(cell_ues)
        self.handover_record = defaultdict(list)
        # we use 100Mbps as the target rate for datarate-fairness
        self.const_rate = 100

    def total_ran(self) -> int:
        return MACRO_CAPACITY + (self.n_cells - 1) * SMALL_CAPACITY

    def insert_ue(self, cid, sid, ue_info):
        self.ue_info[cid][sid][ue_info.ue_id] = ue_info

    def del_ue(self, cid, sid, ue_id):
        self.ue_info[cid][sid].pop(ue_id)

    def cell_num(self, cid):
        n = 0
        for sid in range(self.n_slices):
            n += len(self.ue_info[cid][sid])
        return n

    def slice_num(self, sid):
        n = 0
        for cid in range(self.n_cells):
            n += len(self.ue_info[cid][sid])
        return n

    def cell_slice_df_demand(self, cid, sid):
        total_demand = 0
        for ue_id, ue_info in self.ue_info[cid][sid].items():
            total_demand += self.const_rate / ue_info.get_cqi(cid)
        return total_demand

    def slice_df_demand(self, sid):
        total_demand = 0
        for cid in range(self.n_cells):
            for ue_id, ue_info in self.ue_info[cid][sid].items():
                total_demand += self.const_rate / ue_info.get_cqi(cid)
        return total_demand

    def cell_slice_num(self, cid, sid):
        return len(self.ue_info[cid][sid])

    def cell_relative_demand(self, cid, is_pf: bool = True) -> float:
        """
        The unit of "relative demand" is the RAN percentage.
        The more close of the relative demand to the cell capacity, the more balanced the system is.
        """
        cell_demand = 0
        if is_pf:
            # PF fairness scheduling
            for sid in range(self.n_slices):
                cell_demand += self.cell_slice_num(cid, sid) / self.slice_num(sid)
        else:
            # Datarate fairness scheduling
            for sid in range(self.n_slices):
                cell_demand += self.cell_slice_df_demand(
                    cid, sid
                ) / self.slice_df_demand(sid)
        cell_demand = cell_demand * self.total_ran() / self.n_slices
        return cell_demand

    def ue_relative_demand(self, sid: int) -> float:
        return 1.0 / self.slice_num(sid) * self.total_ran() / self.n_slices

    def ue_df_relative_demand(self, ue: UEInfo, bind_cell: int) -> float:
        return (
            self.total_ran()
            / self.n_slices
            * (self.const_rate / ue.get_cqi(bind_cell))
            / self.slice_df_demand(ue.sid)
        )

    def get_cell_allues(self, cid: int) -> list[UEInfo]:
        all_ues = []
        for sid in range(self.n_slices):
            all_ues = all_ues + list(self.ue_info[cid][sid].values())
        return all_ues

    def log_ue_distribution(self, headlog: str = "") -> None:
        ues_num = np.zeros((self.n_cells, self.n_slices), dtype=float)
        for i in range(self.n_cells):
            for j in range(self.n_slices):
                ues_num[i, j] = self.cell_slice_num(i, j)
        print(headlog + f"ues_num:\n{ues_num}")

    def log_relative_demand(self, headlog: str = "", is_pf: bool = True) -> None:
        ues_demand = np.zeros((self.n_cells, self.n_slices), dtype=float)
        for i in range(self.n_cells):
            for j in range(self.n_slices):
                if is_pf:
                    ues_demand[i, j] = (
                        self.total_ran()
                        / self.n_slices
                        * self.cell_slice_num(i, j)
                        / self.slice_num(j)
                    )
                else:
                    ues_demand[i, j] = (
                        self.total_ran()
                        / self.n_slices
                        * self.cell_slice_df_demand(i, j)
                        / self.slice_df_demand(j)
                    )
        # print(headlog + f"relative_demand:\n{ues_demand}")
        print(headlog + f"cell_demand:\n{np.sum(ues_demand, axis=1)}")

    def log_ran_weights(self, headlog: str = "") -> None:
        print(headlog + f"ran_weights:\n{self.ran_weights}")

    def log_handover_record(self, headlog: str = "") -> None:
        print(headlog + f"ho_record:\n{self.handover_record}")

    def handover_by_num(
        self, cfrom: int, cto: int, n_ues: int, ran_aware: bool
    ) -> None:
        """
        Handover @n_ues users from cell @cfrom to cell @cto with the highest
        channel quality. It's slice agnostic.
        """
        ues_cfrom = self.get_cell_allues(cfrom)
        ues_sorted = sorted(
            ues_cfrom, key=lambda x: -(x.get_cqi(cto) / x.get_cqi(cfrom))
        )
        for i in range(n_ues):
            ue_top = ues_sorted[i]
            if ran_aware and ue_top.get_cqi(cto) / ue_top.get_cqi(cfrom) < 0.5:
                break
            self.ue_info[cfrom][ue_top.sid].pop(ue_top.ue_id)
            self.ue_info[cto][ue_top.sid][ue_top.ue_id] = ue_top
            self.handover_record[(cfrom, cto)].append(ue_top.ue_id)

    def naive_handover(self, ran_aware: bool = False):
        avg_small_num = 0
        for cid in range(self.n_cells):
            avg_small_num += self.cell_num(cid)
        avg_small_num = int(
            SMALL_CAPACITY
            * avg_small_num
            / ((self.n_cells - 1) * SMALL_CAPACITY + MACRO_CAPACITY)
        )
        for cid in range(1, self.n_cells):
            num_cell = self.cell_num(cid)
            if num_cell > avg_small_num:
                self.handover_by_num(cid, 0, num_cell - avg_small_num, ran_aware)
        for cid in range(1, self.n_cells):
            num_cell = self.cell_num(cid)
            if num_cell < avg_small_num:
                self.handover_by_num(0, cid, avg_small_num - num_cell, ran_aware)
        sim.calculate_ran_weights(is_pf=True)

    def handover_by_pf_demand(
        self, cfrom: int, cto: int, demand_offload: float, ran_aware=False
    ):
        """
        Handover by using the number of UEs as the demand
        """
        ues_cfrom = self.get_cell_allues(cfrom)
        ues_sorted = sorted(
            ues_cfrom, key=lambda x: -(x.get_cqi(cto) / x.get_cqi(cfrom))
        )
        while demand_offload > 0:
            ue_top = ues_sorted[0]
            del ues_sorted[0]
            ue_demand = self.ue_relative_demand(ue_top.sid)
            # print(f"ue_id: {ue_top.ue_id}, slice: {ue_top.sid}, ue_demand: {ue_demand}, dmd_offload: {demand_offload}")
            # if the neighbor cell's signal is too bad, stop
            if ran_aware and ue_top.get_cqi(cto) / ue_top.get_cqi(cfrom) < 0.5:
                break
            # if the abs value of offset increase, stop
            if abs(demand_offload) < abs(demand_offload - ue_demand):
                break
            demand_offload -= ue_demand
            self.ue_info[cfrom][ue_top.sid].pop(ue_top.ue_id)
            self.ue_info[cto][ue_top.sid][ue_top.ue_id] = ue_top
            self.handover_record[(cfrom, cto)].append(ue_top.ue_id)

    def handover_by_df_demand(self, cfrom: int, cto: int, ran_aware=False):
        """
        Move UEs from the macro to a small cell or from a small cell to the macro
        Stop when the demand of a small cell meets the capacity
        """
        ues_cfrom = self.get_cell_allues(cfrom)
        ues_sorted = sorted(
            ues_cfrom, key=lambda x: -(x.get_cqi(cto) / x.get_cqi(cfrom))
        )
        from_capacity = MACRO_CAPACITY if cfrom == 0 else SMALL_CAPACITY
        to_capacity = MACRO_CAPACITY if cto == 0 else SMALL_CAPACITY
        to_macro = True if cto == 0 else False
        assert cfrom * cto == 0  # There must be one macro cell
        while True:
            # if any cell is balanced, we just stop
            if to_macro:
                from_demand = self.cell_relative_demand(cfrom, is_pf=False)
                # print(f"ho_df_demand: {cfrom}->{cto}, from_demand: {from_demand}")
                if from_demand <= from_capacity:
                    break
            else:
                to_demand = self.cell_relative_demand(cto, is_pf=False)
                # print(f"ho_df_demand: {cfrom}->{cto}, to_demand: {to_demand}")
                if to_demand >= to_capacity:
                    break
            ue_top = ues_sorted[0]
            del ues_sorted[0]
            # if channel is too bad after handover, break
            if ran_aware and ue_top.get_cqi(cto) / ue_top.get_cqi(cfrom) < 0.5:
                break
            self.ue_info[cfrom][ue_top.sid].pop(ue_top.ue_id)
            self.ue_info[cto][ue_top.sid][ue_top.ue_id] = ue_top
            self.handover_record[(cfrom, cto)].append(ue_top.ue_id)

    def smart_handover_pf(self, ran_aware=False):
        for cid in range(1, self.n_cells):
            cell_demand = self.cell_relative_demand(cid, is_pf=True)
            if cell_demand > SMALL_CAPACITY:
                self.handover_by_pf_demand(
                    cid, 0, cell_demand - SMALL_CAPACITY, ran_aware
                )
        for cid in range(1, self.n_cells):
            cell_demand = self.cell_relative_demand(cid, is_pf=True)
            if cell_demand < SMALL_CAPACITY:
                self.handover_by_pf_demand(
                    0, cid, SMALL_CAPACITY - cell_demand, ran_aware
                )
        # the swap-based ran weight calculation can be suboptimal, and it's np-hard
        # for smart-handover, we directly use relative-demand as the ran weights
        ran_weights = np.zeros((self.n_cells, self.n_slices), dtype=float)
        for cid in range(self.n_cells):
            for sid in range(self.n_slices):
                ran_weights[cid, sid] = (
                    self.cell_slice_num(cid, sid)
                    / self.slice_num(sid)
                    * self.total_ran()
                    / self.n_slices
                )
        if not ran_aware:
            self.ran_weights = ran_weights
        else:
            self.calculate_ran_weights(is_pf=True, init_guess=ran_weights)

    def smart_handover_df(self, ran_aware=False):
        for cid in range(1, self.n_cells):
            cell_demand = self.cell_relative_demand(cid, is_pf=False)
            if cell_demand > SMALL_CAPACITY:
                self.handover_by_df_demand(cid, 0, ran_aware)
        for cid in range(1, self.n_cells):
            cell_demand = self.cell_relative_demand(cid, is_pf=False)
            if cell_demand < SMALL_CAPACITY:
                self.handover_by_df_demand(0, cid, ran_aware)
        ran_weights = np.zeros((self.n_cells, self.n_slices), dtype=float)
        for cid in range(self.n_cells):
            for sid in range(self.n_slices):
                ran_weights[cid, sid] = (
                    self.cell_slice_df_demand(cid, sid)
                    / self.slice_df_demand(sid)
                    * self.total_ran()
                    / self.n_slices
                )
        if not ran_aware:
            self.ran_weights = ran_weights
        else:
            self.calculate_ran_weights(is_pf=False, init_guess=ran_weights)

    def calculate_ran_weights(self, is_pf: bool = True, init_guess=None):
        """
        The current greed approach is suboptimal
        The initial-guess helps to find the optimal result
        """
        init_weights = np.zeros((self.n_cells, self.n_slices), dtype=float)
        if init_guess is None:
            for sid in range(self.n_slices):
                for cid in range(self.n_cells):
                    capacity = MACRO_CAPACITY if cid == 0 else SMALL_CAPACITY
                    init_weights[cid, sid] = capacity / self.n_slices
        else:
            for cid in range(self.n_cells):
                cell_demand = np.sum(init_guess, axis=1)
                cell_capacity = MACRO_CAPACITY if cid == 0 else SMALL_CAPACITY
                if cell_demand[cid] > cell_capacity:
                    init_guess[cid, :] -= (
                        cell_demand[cid] - cell_capacity
                    ) / self.n_slices
                else:
                    init_guess[cid, :] += (
                        cell_capacity - cell_demand[cid]
                    ) / self.n_slices
            init_weights = init_guess
        ideal_weights = np.zeros((self.n_cells, self.n_slices), dtype=float)
        provision = np.zeros((self.n_cells, self.n_slices), dtype=int)
        for sid in range(self.n_slices):
            slice_ran = self.total_ran() / self.n_slices
            for cid in range(self.n_cells):
                if is_pf:
                    ideal_weights[cid, sid] = (
                        slice_ran * self.cell_slice_num(cid, sid) / self.slice_num(sid)
                    )
                else:
                    ideal_weights[cid, sid] = (
                        slice_ran
                        * self.cell_slice_df_demand(cid, sid)
                        / self.slice_df_demand(sid)
                    )
                provision[cid, sid] = provision_metric(
                    init_weights[cid, sid], ideal_weights[cid, sid]
                )
        delta = 0.001
        while True:
            # print(f"provision: \n {provision} ran_weights: \n {ran_weights}")
            swap_once = False
            for lcell in range(self.n_cells):
                for lslice in range(self.n_slices):
                    for rslice in range(self.n_slices):
                        if lslice == rslice:
                            continue
                        for rcell in range(self.n_cells):
                            if lcell == rcell:
                                continue
                            if (
                                provision[lcell][lslice] * provision[lcell][rslice]
                                == -1
                            ):
                                #  skip if both slices are over-provision or under-provision
                                if not (
                                    provision[lcell][lslice] * provision[rcell][lslice]
                                    == -1
                                    and provision[rcell][lslice]
                                    * provision[rcell][rslice]
                                    == -1
                                ):
                                    continue
                                over = provision[lcell][lslice]
                                init_weights[lcell, lslice] -= over * delta
                                init_weights[rcell, lslice] += over * delta
                                init_weights[lcell, rslice] += over * delta
                                init_weights[rcell, rslice] -= over * delta
                                swap_once = True
                                provision[lcell][lslice] = provision_metric(
                                    init_weights[lcell][lslice],
                                    ideal_weights[lcell][lslice],
                                )
                                provision[lcell][rslice] = provision_metric(
                                    init_weights[lcell][rslice],
                                    ideal_weights[lcell][rslice],
                                )
                                provision[rcell][lslice] = provision_metric(
                                    init_weights[rcell][lslice],
                                    ideal_weights[rcell][lslice],
                                )
                                provision[rcell][rslice] = provision_metric(
                                    init_weights[rcell][rslice],
                                    ideal_weights[rcell][rslice],
                                )
            if not swap_once:
                break
        self.ran_weights = init_weights
        # print(f"ran_weights: \n{self.ran_weights}")
        # print(f"provision: \n{provision}")

    def dump_to_csv_pf(self, ofname: str) -> None:
        """
        Calculate the RAN allocation to every UE with PF scheduling
        """
        if self.ran_weights is None:
            raise Exception("The RAN weights are not calculated yet")
        data = {
            "ueid": [],
            "cell": [],
            "slice": [],
            "ran": [],
            "channel": [],
            "posx": [],
            "posy": [],
        }
        for cid in range(self.n_cells):
            for sid in range(self.n_slices):
                ue_subset = self.ue_info[cid][sid]
                if len(ue_subset) == 0:
                    continue
                ran_percentage = self.ran_weights[cid, sid] / len(ue_subset)
                for k, info in ue_subset.items():
                    data["ueid"].append(info.ue_id)
                    data["cell"].append(cid)
                    data["slice"].append(sid)
                    data["ran"].append(ran_percentage)
                    data["channel"].append(info.get_cqi(cid))
                    data["posx"].append(info.coord_x)
                    data["posy"].append(info.coord_y)
        df = pd.DataFrame(data)
        df.to_csv(ofname, index=False)

    def dump_to_csv_df(self, ofname: str) -> None:
        """
        calculate the RAN allocation to every UE with datarate-fair scheduling
        """
        if self.ran_weights is None:
            raise Exception("The RAN weights are not calculated yet")
        data = {"ueid": [], "cell": [], "slice": [], "ran": [], "channel": []}
        for cid in range(self.n_cells):
            for sid in range(self.n_slices):
                ue_subset = self.ue_info[cid][sid]
                if len(ue_subset) == 0:
                    continue
                denominator = 0
                for _, info in ue_subset.items():
                    denominator += 1 / info.get_cqi(cid)
                for _, info in ue_subset.items():
                    data["ueid"].append(info.ue_id)
                    data["cell"].append(cid)
                    data["slice"].append(sid)
                    data["channel"].append(info.get_cqi(cid))
                    data["ran"].append(
                        self.ran_weights[cid, sid] * 1 / info.get_cqi(cid) / denominator
                    )
        df = pd.DataFrame(data)
        df.to_csv(ofname, index=False)


def gen_slice_ue(rand_seed):
    random.seed(rand_seed)
    ues_num = np.zeros((N_CELLS, N_SLICES), dtype=int)
    for i in range(N_CELLS):
        for j in range(N_SLICES):
            if i == 0:
                ues_num[i, j] = random.randint(0, MAX_RAND_UE * MACRO_CAPACITY)
            else:
                ues_num[i, j] = random.randint(0, MAX_RAND_UE * SMALL_CAPACITY)
    return ues_num


# def gen_screw_slice_ue(rand_seed):
#     random.seed(rand_seed)
#     slice_ues = [20, 20, 40, 60, 60]
#     ues_num = np.zeros((N_CELLS, N_SLICES), dtype=int)
#     for i in range(N_SLICES):
#         points = sorted([random.randint(0, slice_ues[i]) for _ in range(N_CELLS + MACRO_WEIGHT - 2)])
#         points.append(slice_ues[i])
#         for j in range(N_CELLS):
#             if j == 0:
#                 ues_num[j, i] = points[MACRO_WEIGHT - 1]
#             else:
#                 ues_num[j, i] = points[MACRO_WEIGHT + j - 1] - points[MACRO_WEIGHT + j - 2]
#     return ues_num


def gen_screw_slice_ue(rand_seed):
    random.seed(rand_seed)
    slice_min_ues = [1, 1, 1, 10, 10]
    slice_max_ues = [5, 5, 5, 20, 20]
    # slice_min_ues = [5, 5, 5, 10, 10]
    # slice_max_ues = [10, 10, 10, 20, 20]
    ues_num = np.zeros((N_CELLS, N_SLICES), dtype=int)
    for i in range(N_CELLS):
        for j in range(N_SLICES):
            if i == 0:
                ues_num[i, j] = random.randint(
                    slice_min_ues[j] * MACRO_CAPACITY, slice_max_ues[j] * MACRO_CAPACITY
                )
            else:
                ues_num[i, j] = random.randint(
                    slice_min_ues[j] * SMALL_CAPACITY, slice_max_ues[j] * SMALL_CAPACITY
                )
    return ues_num


def init_simulator(ues_num, rand_seed):
    n_cells = len(ues_num)
    n_slices = len(ues_num[0])
    cell_all_ues = [[] for i in range(n_cells)]
    with open("./five_cell.csv", "r") as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            ue_id = int(row[0])
            cell_rate = []
            for i in range(n_cells):
                cell_rate.append([int(row[2 * i + 1]), float(row[2 * i + 2])])
            sort_by_cell = sorted(cell_rate, key=lambda x: x[0])
            ue_info = UEInfo(ue_id, sort_by_cell, int(row[-2]), int(row[-1]))
            sort_by_rate = sorted(cell_rate, key=lambda x: -x[1])
            key_cell = sort_by_rate[0][0]
            if key_cell >= n_cells:
                continue
            cell_all_ues[key_cell].append(ue_info)
    sim = Simulator(n_cells, n_slices)
    random.seed(rand_seed)
    for cid in range(n_cells):
        cell_ue_num = sum(ues_num[cid])
        cell_ues = []
        # try to make UEs belonging to different neighbor cells same
        if cid == 0:
            macro_ues = deepcopy(cell_all_ues[0])
            max_neighbor_cell = int(cell_ue_num / (n_cells - 1)) + 1
            neighbor_cell_counter = defaultdict(int)
            for i, ue in enumerate(macro_ues):
                cqi_all = ue.cqi_all
                sort_by_rate = sorted(cqi_all, key=lambda x: -x[1])
                assert sort_by_rate[0][0] == 0
                neighbor_cell = sort_by_rate[1][0]
                if neighbor_cell_counter[neighbor_cell] < max_neighbor_cell:
                    cell_ues.append(ue)
                    neighbor_cell_counter[neighbor_cell] += 1
                if len(cell_ues) == cell_ue_num:
                    break
            if len(cell_ues) != cell_ue_num:
                print(f"{cell_ues}, {cell_ue_num}")
                raise Exception("Not enough macro-cell UE selected for experiments")
        else:
            cell_ues = random.sample(cell_all_ues[cid], cell_ue_num)
        ueid = 0
        for sid in range(n_slices):
            for k in range(ues_num[cid][sid]):
                cell_ues[ueid].set_sid(sid)
                sim.insert_ue(cid, sid, cell_ues[ueid])
                ueid += 1
    return sim


N_SLICES = 5
N_CELLS = 5
MAX_RAND_UE = 10
MACRO_CAPACITY = 2
SMALL_CAPACITY = 1
CASENAME = "macro_"
is_pf_schedule = True
n_samples = 20

if is_pf_schedule:
    ODIR = "./sample_data/"
    for i in range(n_samples):
        print(f"\nrandom seed: {i}")
        # ues_num = gen_screw_slice_ue(i)
        ues_num = gen_slice_ue(i)
        try:
            sim = init_simulator(ues_num, i)
        except ValueError as ve:
            print(f"rand_seed {i} doesn't work")
            continue
        sim.calculate_ran_weights()
        sim.log_ue_distribution("Origin, ")
        sim.log_relative_demand("Origin, ")
        sim.dump_to_csv_pf(ODIR + "origin_" + CASENAME + str(i) + ".csv")

        sim.naive_handover()
        sim.log_relative_demand("NaiveHO, ")
        sim.dump_to_csv_pf(ODIR + "naiveho_" + CASENAME + str(i) + ".csv")
        sim.log_handover_record("NaiveHO, ")

        sim = init_simulator(ues_num, i)
        sim.naive_handover(ran_aware=True)
        sim.dump_to_csv_pf(ODIR + "naiveawareho_" + CASENAME + str(i) + ".csv")
        # sim.log_handover_record("NaiveHO, ")

        sim = init_simulator(ues_num, i)
        sim.smart_handover_pf(False)
        sim.log_relative_demand("UnawareHO, ")
        sim.log_ran_weights("UnawareHO: ")
        sim.dump_to_csv_pf(ODIR + "unawareho_" + CASENAME + str(i) + ".csv")
        sim.log_handover_record("UnawareHO, ")

        sim = init_simulator(ues_num, i)
        sim.smart_handover_pf(True)
        sim.log_relative_demand("AwareHO: ")
        sim.log_ran_weights("AwareHO: ")
        sim.dump_to_csv_pf(ODIR + "awareho_" + CASENAME + str(i) + ".csv")
        sim.log_handover_record("AwareHO, ")

if not is_pf_schedule:
    ODIR = "./sample_data_df/"
    for i in range(n_samples):
        print(f"\nrandom seed: {i}")
        # ues_num = gen_screw_slice_ue(i)
        ues_num = gen_slice_ue(i)
        try:
            sim = init_simulator(ues_num, i)
        except ValueError as ve:
            print(f"rand_seed {i} doesn't work")
            continue
        sim.calculate_ran_weights()
        sim.log_relative_demand("Origin: ", is_pf=False)
        sim.dump_to_csv_df(ODIR + "origin_" + CASENAME + str(i) + ".csv")

        sim.naive_handover()
        sim.log_relative_demand("NaiveHO: ", is_pf=False)
        sim.dump_to_csv_df(ODIR + "naiveho_" + CASENAME + str(i) + ".csv")
        sim.log_handover_record("NaiveHO, ")

        sim = init_simulator(ues_num, i)
        sim.smart_handover_df(ran_aware=False)
        sim.log_relative_demand("UnawareHO: ", is_pf=False)
        sim.dump_to_csv_df(ODIR + "unawareho_" + CASENAME + str(i) + ".csv")
        sim.log_handover_record("UnawareHO, ")
