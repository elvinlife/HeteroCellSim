#!/usr/bin/python3
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from random import randint


class UEInfo:
    def __init__(self, ue_id, cqi_all):
        """
        self.cqi_all: [[0, datarate_c0], [1, datarate_c1], [2, datarate_c2]]
        """
        self.ue_id = ue_id
        self.cqi_all = cqi_all

    def __str__(self) -> str:
        ret_str = "("
        for cqi in self.cqi_all:
            ret_str = ret_str + str(cqi[1]) + ","
        ret_str = ret_str + ")"
        return ret_str

    def set_weight(self, weight):
        self.weight = weight

    def set_sid(self, sid):
        self.sid = sid

    def get_cqi(self, cid):
        return self.cqi_all[cid][1]


class Simulator:
    def __init__(self, n_cells: int, n_slices: int) -> None:
        """
        self.cell_ues: store the information of all users in every cell -- each cell stores k&v values
        self.user_cell: store the current cell id of every user
        """
        self.n_cells = n_cells
        self.n_slices = n_slices
        self.cell_ues = []
        for i in range(self.n_cells):
            self.cell_ues.append({})
        self.user_cell = {}

    def total_ran(self) -> int:
        return MACRO_CAPACITY + (self.n_cells - 1) * SMALL_CAPACITY

    def insert_ue_cell(self, cid, ue_info):
        self.cell_ues[cid][ue_info.ue_id] = ue_info

    def remove_ue_cell(self, cid, ue_id):
        self.cell_ues[cid].pop(ue_id)

    def get_cell_util(self, cells):
        util = 0
        for cell in cells:
            for _, ue in self.cell_ues[cell].items():
                util += ue.weight * np.log(self.get_ue_metric(cell, ue))
        return util

    def get_ue_metric(self, cid, ue_info, is_tp=True):
        """
        If the user is in @cid cell, we calculate the TP
        Else, we insert it first and remove before exiting
        """
        should_remove = False
        ue_id = ue_info.ue_id
        if ue_id not in self.cell_ues[cid]:
            self.cell_ues[cid][ue_id] = ue_info
            should_remove = True
        total_weight = 0
        for k, ue in self.cell_ues[cid].items():
            total_weight += ue.weight
        ratio = ue_info.weight / total_weight
        capacity = MACRO_CAPACITY if cid == 0 else SMALL_CAPACITY
        if should_remove:
            self.cell_ues[cid].pop(ue_id)
        if is_tp:
            return ratio * capacity * ue_info.get_cqi(cid)
        else:
            return ratio * capacity

    def maxtp_move(self, cfroms):
        """
        prev_cell: if move, the previous cell of the moved UE
        start_cell: if move, the new cell of the moved UE
        """
        max_tp_ratio = 0
        ueinfo_star = None
        star_cell = -1
        prev_cell = -1
        for cfrom in cfroms:
            for _, ueinfo in self.cell_ues[cfrom].items():
                current_tp = self.get_ue_metric(cfrom, ueinfo)
                for cid in range(self.n_cells):
                    if cid != cfrom:
                        tp = self.get_ue_metric(cid, ueinfo)
                        if tp / current_tp > max_tp_ratio:
                            max_tp_ratio = tp / current_tp
                            ueinfo_star = ueinfo
                            star_cell = cid
                            prev_cell = cfrom
        return (max_tp_ratio, ueinfo_star, star_cell, prev_cell)

    def maxutil_move(self, cfroms):
        max_util_ratio = 0
        ueinfo_star = None
        star_cell = -1
        prev_cell = -1
        for cfrom in cfroms:
            for _, ueinfo in list(self.cell_ues[cfrom].items()):
                for cid in range(self.n_cells):
                    util_origin = self.get_cell_util([cfrom, cid])
                    if cid != cfrom:
                        self.remove_ue_cell(cfrom, ueinfo.ue_id)
                        self.insert_ue_cell(cid, ueinfo)
                        util_after = self.get_cell_util([cfrom, cid])
                        self.remove_ue_cell(cid, ueinfo.ue_id)
                        self.insert_ue_cell(cfrom, ueinfo)
                        if util_after / util_origin > max_util_ratio:
                            max_util_ratio = util_after / util_origin
                            ueinfo_star = ueinfo
                            star_cell = cid
                            prev_cell = cfrom
        return (max_util_ratio, ueinfo_star, star_cell, prev_cell)

    def insert_ue(self, ue_info):
        max_tp = 0
        insert_cell = -1
        # insert the UE to the cell that gens max tp
        for cid in range(self.n_cells):
            tp = self.get_ue_metric(cid, ue_info)
            if tp > max_tp:
                max_tp = tp
                insert_cell = cid
        self.insert_ue_cell(insert_cell, ue_info)
        assert insert_cell != -1
        # move a UE from insert_cell that may achieve higher TP in another cell
        (max_tp_ratio, ueinfo_star, star_cell, prev_cell) = self.maxtp_move(
            [insert_cell]
        )
        if max_tp_ratio < 1:
            return
        # let's move a new user
        self.remove_ue_cell(prev_cell, ueinfo_star.ue_id)
        self.insert_ue_cell(star_cell, ueinfo_star)
        # three more moves
        for i in range(2):
            (max_tp_ratio, ueinfo_star, star_cell, prev_cell) = self.maxtp_move(
                [prev_cell, star_cell]
            )
            if max_tp_ratio < 1:
                return
            self.remove_ue_cell(prev_cell, ueinfo_star.ue_id)
            self.insert_ue_cell(star_cell, ueinfo_star)
        (max_util_ratio, ueinfo_star, star_cell, prev_cell) = self.maxutil_move(
            [prev_cell, star_cell]
        )
        if max_util_ratio < 1:
            return
        self.remove_ue_cell(prev_cell, ueinfo_star.ue_id)
        self.insert_ue_cell(star_cell, ueinfo_star)

    def log_ue_distribution(self):
        print("ue_distribution: ")
        for cid in range(self.n_cells):
            ues = self.cell_ues[cid].keys()
            print(f"{ues}")

    def dump_to_csv(self, ofname: str) -> None:
        data = {"ueid": [], "cell": [], "slice": [], "ran": [], "channel": []}
        cell_tune = 0
        macro_alloc = defaultdict(float)
        slice_ues = defaultdict(int)
        for cid in range(0, self.n_cells):
            if cid == cell_tune:
                continue
            for _, ue_info in self.cell_ues[cid].items():
                data["ueid"].append(ue_info.ue_id)
                data["cell"].append(cid)
                data["slice"].append(ue_info.sid)
                data["channel"].append(ue_info.get_cqi(cid))
                ran_quota = self.get_ue_metric(cid, ue_info, False)
                data["ran"].append(ran_quota)
                macro_alloc[ue_info.sid] += ran_quota
        # tune the RAN alloc in the macro cell
        for sid in range(self.n_slices):
            macro_alloc[sid] = self.total_ran() / self.n_slices - macro_alloc[sid]
        for _, ue_info in self.cell_ues[cell_tune].items():
            slice_ues[ue_info.sid] += 1
        for _, ue_info in self.cell_ues[cell_tune].items():
            data["ueid"].append(ue_info.ue_id)
            data["cell"].append(cell_tune)
            data["slice"].append(ue_info.sid)
            data["channel"].append(ue_info.get_cqi(cell_tune))
            data["ran"].append(macro_alloc[ue_info.sid] / slice_ues[ue_info.sid])
            # data["ran"].append(self.get_ue_metric(cell_tune, ue_info, False))
        df = pd.DataFrame(data)
        df.to_csv(ofname, index=False)


def gen_slice_ue(rand_seed):
    random.seed(rand_seed)
    ues_num = np.zeros((N_CELLS, N_SLICES), dtype=int)
    for i in range(N_CELLS):
        for j in range(N_SLICES):
            if i == 0:
                ues_num[i, j] = random.randint(0, MAX_RAND_UE * MACRO_UES)
            else:
                ues_num[i, j] = random.randint(0, MAX_RAND_UE * SMALL_UES)
    return ues_num


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
                    slice_min_ues[j] * MACRO_UES, slice_max_ues[j] * MACRO_UES
                )
            else:
                ues_num[i, j] = random.randint(
                    slice_min_ues[j] * SMALL_UES, slice_max_ues[j] * SMALL_UES
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
            sort_by_rate = sorted(cell_rate, key=lambda x: -x[1])
            key_cell = sort_by_rate[0][0]
            if key_cell >= n_cells:
                continue
            sort_by_cell = sorted(cell_rate, key=lambda x: x[0])
            ue_info = UEInfo(ue_id, sort_by_cell)
            cell_all_ues[key_cell].append(ue_info)
    sim = Simulator(n_cells, n_slices)
    random.seed(rand_seed)
    for cid in range(n_cells):
        cell_ue_num = sum(ues_num[cid])
        # @cell_ues stores randomly sampled users, and is yet to be assigned slice id
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
        slice_uenum = np.sum(ues_num, axis=0)
        for sid in range(n_slices):
            slice_ue = slice_uenum[sid] + randint(-10, 10)
            for k in range(ues_num[cid][sid]):
                cell_ues[ueid].set_sid(sid)
                cell_ues[ueid].set_weight(1 / slice_ue)
                # cell_ues[ueid].set_weight(1)
                sim.insert_ue(cell_ues[ueid])
                ueid += 1
    return sim


def print_ranalloc():
    ODIR = "./sample_data/"
    for i in range(10):
        fname = ODIR + "mora_" + CASENAME + str(i) + ".csv"
        df = pd.read_csv(fname)
        slices_ran = []
        for sid in range(N_SLICES):
            slice_ran = 0
            df_slice = df[df["slice"] == sid]
            for _, row in df_slice.iterrows():
                slice_ran += row["ran"]
            slices_ran.append(slice_ran)
        print(f"slice: {slices_ran}")
        cells_ran = []
        for cid in range(N_CELLS):
            cell_ran = 0
            df_cell = df[df["cell"] == cid]
            for _, row in df_cell.iterrows():
                cell_ran += row["ran"]
            cells_ran.append(cell_ran)
        print(f"cell: {cells_ran}")


N_SLICES = 5
N_CELLS = 5
MAX_RAND_UE = 10
MACRO_CAPACITY = 5
SMALL_CAPACITY = 1
MACRO_UES = 5
SMALL_UES = 1
CASENAME = "macro_"
is_pf_schedule = True
n_samples = 20


def gen_all_results():
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
            sim.log_ue_distribution()
            sim.dump_to_csv(ODIR + "mora_" + CASENAME + str(i) + ".csv")


gen_all_results()
print_ranalloc()
