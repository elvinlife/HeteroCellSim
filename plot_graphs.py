#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

COLORS = ["#134f5c", "#351c75", "#990000", "#b45f06", "#bf9000", "#38761d"]


def plot_perue_bw(ifname, ofname):
    df = pd.read_csv(ifname)
    n_slice = 3
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_slice):
        slice_ues = df[df["slice"] == i]
        ue_datarate = []
        for _, row in slice_ues.iterrows():
            ue_datarate.append(float(row["ran"]) * float(row["channel"]))
        data_sort = np.sort(ue_datarate)
        cdf_array = np.arange(1, (len(data_sort) + 1)) / len(data_sort)
        ax.plot(cdf_array, data_sort, label="Slice" + str(i))
    ax.set_xlabel("Percentage")
    ax.set_ylabel("Datarate(Mbps)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ofname)


def plot_perue_ran(ifname, ofname):
    df = pd.read_csv(ifname)
    n_slice = 3
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_slice):
        slice_ues = df[df["slice"] == i]
        ue_datarate = []
        for _, row in slice_ues.iterrows():
            ue_datarate.append(float(row["ran"]) * 100)
        data_sort = np.sort(ue_datarate)
        cdf_array = np.arange(1, (len(data_sort) + 1)) / len(data_sort)
        ax.plot(cdf_array, data_sort, label="Slice" + str(i))
    ax.set_xlabel("Percentage")
    ax.set_ylabel("RAN(%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ofname)


def plot_scheme_compare(sys_ids: list, seed_id: int, ofname: str, is_ran: True):
    """
    Plot per-UE ran&data_rate comparison across multiple schemes
    """
    dfs = []
    fnames = [SDIR + PREFIXES[id] + str(seed_id) + ".csv" for id in sys_ids]
    labels = [LABELS[id] for id in sys_ids]
    if not os.path.exists(fnames[0]):
        return
    for fname in fnames:
        dfs.append(pd.read_csv(fname))
    n_slices = dfs[0]["slice"].max() + 1
    fig, ax = plt.subplots(figsize=(20, 5))
    bar_width = 0.3
    for i, df in enumerate(dfs):
        ue_begin = 0
        for sid in range(n_slices):
            slice_ues = df[df["slice"] == sid]
            slice_ue_metric = []
            for _, row in slice_ues.iterrows():
                if is_ran:
                    slice_ue_metric.append(float(row["ran"]) * 100)
                else:
                    slice_ue_metric.append(float(row["ran"]) * float(row["channel"]))
            data_sort = sorted(slice_ue_metric)
            x_array = np.arange(ue_begin, ue_begin + len(data_sort)) + bar_width * (
                i - 1
            )
            ax.bar(
                x_array, data_sort, width=bar_width, label=labels[i] + "-S" + str(sid)
            )
            ue_begin += len(data_sort)
    ax.set_xlabel("UE ID")
    if is_ran:
        ax.set_ylabel("RAN percentage")
    else:
        ax.set_ylabel("Throughput(Mbps)")
    ax.set_ylim((0, 20))
    ax.legend()
    plt.tight_layout()
    print(ofname)
    plt.savefig(ofname)


def get_sublist(l: list, lowb: float, upperb: float) -> list:
    return l[int(lowb * len(l)) : int(upperb * len(l))]


def get_ran_improve(f1: str, f2: str, lowb: float, upperb: float):
    """
    get the RAN improvement for every UE
    """
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    n_slices = df1["slice"].max() + 1
    ran1_array = []
    ran2_array = []
    improve_array = []
    for sid in range(n_slices):
        s1_ues = sorted(list(df1[df1["slice"] == sid]["ran"]))
        s2_ues = sorted(list(df2[df2["slice"] == sid]["ran"]))
        ran1_array = ran1_array + get_sublist(s1_ues, lowb, upperb)
        ran2_array = ran2_array + get_sublist(s2_ues, lowb, upperb)
    for i in range(len(ran1_array)):
        improve_array.append((ran2_array[i] / ran1_array[i] - 1))
    return sorted(improve_array)


def get_ran_maxmin(fname: str) -> list:
    """
    For every slice, get the ratio between the RAN of the UE with maximum RAN
    and the UE with minimal RAN
    """
    df = pd.read_csv(fname)
    n_slices = df["slice"].max() + 1
    scheme1_array = []
    for sid in range(n_slices):
        s1_ues = sorted(list(df[df["slice"] == sid]["ran"]))
        scheme1_array.append(s1_ues[-1] / s1_ues[0])
    return scheme1_array


def get_metrics(fname: str, lowb: float, upperb: float, is_ran: True):
    """
    For the csv, sort the UEs by the metric in ascending order,
    return the metrics in range lowb ~ upperb
    """
    df = pd.read_csv(fname)
    n_slices = df["slice"].max() + 1
    allue_metrics = []
    for sid in range(n_slices):
        slice_metrics = []
        slice_ues = df[df["slice"] == sid]
        for _, row in slice_ues.iterrows():
            metric = row["ran"] if is_ran else row["ran"] * row["channel"]
            slice_metrics.append(metric)
        slice_metrics = get_sublist(sorted(slice_metrics), lowb, upperb)
        allue_metrics = allue_metrics + slice_metrics
    return sorted(allue_metrics)


def get_avg_metric(fname: str, lowb: float, upperb: float, is_ran: True) -> list:
    """
    Get the average metric of UEs with RAN in [lowb, upperb]
    """
    df = pd.read_csv(fname)
    n_slices = df["slice"].max() + 1
    slice_avg_metrics = []
    for sid in range(n_slices):
        slice_metrics = []
        slice_ues = df[df["slice"] == sid]
        for _, row in slice_ues.iterrows():
            slice_metrics.append((row["ueid"], row["ran"], row["ran"] * row["channel"]))
        slice_metrics = get_sublist(
            sorted(slice_metrics, key=lambda x: x[1]), lowb, upperb
        )
        slice_tail_metrics = []
        for metric in slice_metrics:
            if is_ran:
                slice_tail_metrics.append(metric[1])
            else:
                slice_tail_metrics.append(metric[2])
        slice_avg_metrics.append(np.average(slice_tail_metrics))
    return slice_avg_metrics


def get_avg_lowtp(fname: str, lowb: float, upperb: float) -> list:
    """
    Get the average throughput of UEs with TP in [lowb, upperb]
    """
    df = pd.read_csv(fname)
    n_slices = df["slice"].max() + 1
    slice_avg_metrics = []
    for sid in range(n_slices):
        slice_metrics = []
        slice_ues = df[df["slice"] == sid]
        for _, row in slice_ues.iterrows():
            slice_metrics.append((row["ueid"], row["ran"] * row["channel"]))
        slice_metrics = get_sublist(
            sorted(slice_metrics, key=lambda x: x[1]), lowb, upperb
        )
        slice_tail_metrics = []
        for metric in slice_metrics:
            slice_tail_metrics.append(metric[1])
        slice_avg_metrics.append(np.average(slice_tail_metrics))
    return slice_avg_metrics


def get_sameue_improve(f1: str, f2: str, lowb: float, upperb: float, is_ran: True):
    """
    Here, f1 is the baseline, we sort the UEs in f1 by the metric, and record
    the UEs in range lowb ~ upperb. Then we check the performance improvement of
    these users in f2
    """
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    n_slices = df1["slice"].max() + 1
    improve_array = []
    for sid in range(n_slices):
        s1_ues = df1[df1["slice"] == sid]
        ue1_metrics = []
        ue1_metrics_kv = {}
        for _, row in s1_ues.iterrows():
            ue_id = row["ueid"]
            metric = row["ran"] if is_ran else row["ran"] * row["channel"]
            ue1_metrics.append((ue_id, metric))
        ue1_metrics = sorted(ue1_metrics, key=lambda x: x[1])
        ue1_metrics = get_sublist(ue1_metrics, lowb, upperb)
        for item in ue1_metrics:
            ue1_metrics_kv[item[0]] = item[1]
        slice_improve = []
        s2_ues = df2[df2["slice"] == sid]
        for _, row in s2_ues.iterrows():
            ue_id = row["ueid"]
            if ue_id in ue1_metrics_kv:
                metric = row["ran"] if is_ran else row["ran"] * row["channel"]
                slice_improve.append(metric / ue1_metrics_kv[ue_id] - 1)
                # if metric / ue1_metrics_kv[ue_id] - 1 < -0.5:
                #     print(ue1_metrics)
                #     raise Exception(f"file: {f1} ue: {ue_id}")
        improve_array = improve_array + slice_improve
    return sorted(improve_array)


def get_relademand(ifname: str, is_pf: bool = True):
    """
    For one trace, get the relative demand of every cell to see how balanced
    the allocation strategy is
    """
    df = pd.read_csv(ifname)
    n_slices = df["slice"].max() + 1
    n_cells = df["cell"].max() + 1
    slices_demand = [0 for i in range(n_slices)]
    cells_demand = [0 for i in range(n_cells)]
    slice_ran = TOTAL_RAN / n_slices
    for sid in range(n_slices):
        slice_ues = df[df["slice"] == sid]
        if is_pf:
            slices_demand[sid] = len(slice_ues)
        else:
            for _, row in slice_ues.iterrows():
                slices_demand[sid] += 1 / row["channel"]
    for cid in range(n_cells):
        for sid in range(n_slices):
            specific_ues = df[df["slice"] == sid]
            specific_ues = specific_ues[specific_ues["cell"] == cid]
            if is_pf:
                cells_demand[cid] += len(specific_ues) / slices_demand[sid] * slice_ran
            else:
                for _, row in specific_ues.iterrows():
                    cells_demand[cid] += (
                        (1 / row["channel"]) / slices_demand[sid] * slice_ran
                    )
    return cells_demand


def plot_relademand_cdf(prefixes: list, labels: list, ofname: str, is_pf: bool = True):
    line_styles = ["solid", "dashed", "dotted", "dashed", "dotted"]
    fig, ax = plt.subplots(figsize=(8, 6))
    scheme_metrics = [[] for i in range(len(prefixes))]
    for i in range(N_SAMPLES):
        if not os.path.exists(SDIR + prefixes[0] + str(i) + ".csv"):
            continue
        for j, prefix in enumerate(prefixes):
            cells_demand = get_relademand(SDIR + prefix + str(i) + ".csv", is_pf)
            scheme_metrics[j] = scheme_metrics[j] + cells_demand
    for j, label in enumerate(labels):
        ax.ecdf(scheme_metrics[j], label=label, linestyle=line_styles[j])
    ax.set_xlabel("Relative Demand")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ofname)


def plot_absvalue_multi_sys(
    prefixes: list,
    labels: list,
    ofname: str,
    is_ran: bool,
    lowb: float,
    upperb: float,
    sort_by_ran: bool = True,
):
    line_styles = ["solid", "dashed", "dotted", "dashed", "dotted", "dashed"]
    default_font = 18
    fig, ax = plt.subplots(figsize=(8, 6))
    scheme_metrics = [[] for i in range(len(prefixes))]
    for i in range(N_SAMPLES):
        if not os.path.exists(SDIR + prefixes[0] + str(i) + ".csv"):
            continue
        for j, prefix in enumerate(prefixes):
            if sort_by_ran:
                metrics = get_avg_metric(
                    SDIR + prefix + str(i) + ".csv", lowb, upperb, is_ran
                )
            else:
                metrics = get_avg_lowtp(SDIR + prefix + str(i) + ".csv", lowb, upperb)
            scheme_metrics[j] = scheme_metrics[j] + metrics
    for j, label in enumerate(labels):
        # print(f"{label}: median {np.percentile(scheme_metrics[j], 90)}")
        ax.ecdf(
            scheme_metrics[j],
            label=label,
            linestyle=line_styles[j],
            color=COLORS[j],
            linewidth=2.5,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=default_font + 2)
    if sort_by_ran and is_ran:
        ax.set_xlabel("RAN allocation", fontsize=default_font + 2)
    else:
        ax.set_xlabel("Throughput(Mbps)", fontsize=default_font + 2)
    ax.set_ylabel("Cumulative Probability", fontsize=default_font + 2)
    ax.tick_params(axis="both", labelsize=default_font)
    ax.grid(axis="both", alpha=0.2)
    plt.tight_layout()
    fig.savefig(ofname + FTYPE)


def plot_improve_multi_sys(
    basic_id: int,
    prefixes: list,
    labels: list,
    ofname: str,
    is_ran: bool,
    lowb: float,
    upperb: float,
):
    """
    Plot the CDF graph of the metric improvement from sys1 to sys2 and sys3
    """
    line_styles = ["solid", "dashed", "dotted", "dashed", "dotted"]
    fig, ax = plt.subplots(figsize=(8, 6))
    all_improves = [[] for i in range(len(prefixes))]
    for i in range(N_SAMPLES):
        if not os.path.exists(SDIR + prefixes[basic_id] + str(i) + ".csv"):
            continue
        for j, prefix in enumerate(prefixes):
            improve = get_sameue_improve(
                SDIR + PREFIXES[basic_id] + str(i) + ".csv",
                SDIR + prefix + str(i) + ".csv",
                lowb,
                upperb,
                is_ran,
            )
            all_improves[j] = all_improves[j] + improve
    max_value = 0
    for j, label in enumerate(labels):
        if j != basic_id:
            all_improves[j] = sorted(all_improves[j])
            ax.ecdf(
                all_improves[j],
                label=labels[basic_id] + "->" + label,
                linestyle=line_styles[j],
            )
            if all_improves[j][-1] > max_value:
                max_value = all_improves[j][-1]
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(loc="lower right")
    if max_value > 4:
        ax.set_xlim(right=4)
    if is_ran:
        ax.set_xlabel("RAN-Improve")
    else:
        ax.set_xlabel("Throughput-Improve")
    ax.set_ylabel("Probability")
    plt.tight_layout()
    fig.savefig(ofname)


def plot_slice_improve(
    sys0: int,
    baselines: list,
    ofname: str,
    is_ran: bool,
    lowb: float,
    upperb: float,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    max_improve_basic = [[] for i in range(len(baselines))]
    min_improve_basic = [[] for i in range(len(baselines))]
    for i in range(N_SAMPLES):
        if not os.path.exists(SDIR + PREFIXES[0] + str(i) + ".csv"):
            continue
        baseline_metrics = []
        oursys_metrics = get_avg_metric(
            SDIR + PREFIXES[sys0] + str(i) + ".csv", lowb, upperb, is_ran
        )
        for base_id in baselines:
            baseline = get_avg_metric(
                SDIR + PREFIXES[base_id] + str(i) + ".csv", lowb, upperb, is_ran
            )
            baseline_metrics.append(baseline)
        for k, baseline in enumerate(baseline_metrics):
            basic_improve = []
            for j, metric in enumerate(oursys_metrics):
                basic_improve.append(metric / baseline[j] - 1)
            max_improve_basic[k].append(max(basic_improve))
            min_improve_basic[k].append(min(basic_improve))

    x_array = np.arange(8)
    x_ticklabel = [
        "max_origin",
        "min_origin",
        "max_naiveho",
        "min_naiveho",
        "max_isolateho",
        "min_isolateho",
        "max_mora",
        "min_mora",
    ]
    y_array = []
    ystddev_array = []
    print(len(max_improve_basic))
    for i in range(len(baselines)):
        y_array.append(np.mean(max_improve_basic[i]))
        y_array.append(np.mean(min_improve_basic[i]))
        ystddev_array.append(np.std(max_improve_basic[i]))
        ystddev_array.append(np.std(min_improve_basic[i]))
    ax.bar(x_array, y_array, width=0.3)
    ax.errorbar(
        x_array,
        y_array,
        yerr=ystddev_array,
        elinewidth=1,
        capsize=10,
        fmt=".",
        color="k",
    )
    ax.set_ylim(bottom=-0.1)
    ax.set_xticks(x_array)
    ax.set_xticklabels(x_ticklabel)
    ax.set_ylabel("Improvement")
    plt.tight_layout()
    fig.savefig(ofname)


def plot_maxmin_ratio(sys1: int, sys2: int, sys3: int, ofname: str):
    fig, ax = plt.subplots(figsize=(20, 6))
    n_samples = 10
    scheme1_ratio = []
    scheme2_ratio = []
    scheme3_ratio = []
    width = 0.25
    for i in range(n_samples):
        if not os.path.exists(SDIR + PREFIXES[sys1] + str(i) + ".csv"):
            continue
        scheme1_array = get_ran_maxmin(SDIR + PREFIXES[sys1] + str(i) + ".csv")
        scheme2_array = get_ran_maxmin(SDIR + PREFIXES[sys2] + str(i) + ".csv")
        scheme3_array = get_ran_maxmin(SDIR + PREFIXES[sys3] + str(i) + ".csv")
        scheme1_ratio = scheme1_ratio + scheme1_array
        scheme2_ratio = scheme2_ratio + scheme2_array
        scheme3_ratio = scheme3_ratio + scheme3_array
    ax.bar(
        np.arange(len(scheme1_ratio)) - width,
        scheme1_ratio,
        width=width,
        label=LABELS[sys1],
    )
    ax.bar(
        np.arange(len(scheme2_ratio)), scheme2_ratio, width=width, label=LABELS[sys2]
    )
    ax.bar(
        np.arange(len(scheme3_ratio)) + width,
        scheme3_ratio,
        width=width,
        label=LABELS[sys3],
    )
    ax.legend()
    ax.set_xlabel("Slice-ID")
    ax.set_ylabel("Max-Min Ratio")
    fig.savefig(ofname)


is_pf_schedule = True
if is_pf_schedule:
    SDIR = "./sample_data/"
    FIGDIR = "./figures/"
    # LABELS = ["Origin", "NaiveUnaware", "NaiveHO", "SmartUnaware", "SmartHO", "IsolateHO"]
    # PREFIXES = ["origin_", "naiveunaware_", "naiveho_", "smartunaware_", "smartho_", "isolateho_"]
    LABELS = ["Origin", "NaiveHO", "SmartHO", "IsolateHO", "MORA"]
    PREFIXES = ["origin_", "naiveho_", "smartho_", "isolateho_", "mora_"]
else:
    SDIR = "./sample_data_df/"
    FIGDIR = "./figures-df/"
    LABELS = ["Origin", "NaiveHO", "UnawareHO"]
    PREFIXES = ["origin_", "naiveho_", "unawareho_"]

N_SAMPLES = 20
CASE_NAME = "macro_"
TOTAL_RAN = 9
FTYPE = ".png"

PREFIXES = [prefix + CASE_NAME for prefix in PREFIXES]

# for i in range(2):
#     plot_scheme_compare(
#         [3, 4], i, FIGDIR + CASE_NAME + "compare_ran" + str(i) + ".png", True
#     )
#     plot_scheme_compare(
#         [3, 4], i, FIGDIR + CASE_NAME + "compare_datarate" + str(i) + ".png", False
#     )

plot_slice_improve(
    2, [0, 1, 3, 4], FIGDIR + CASE_NAME + "ran_improve.png", True, 0, 0.2
)

plot_slice_improve(
    2, [0, 1, 3, 4], FIGDIR + CASE_NAME + "tp_improve.png", False, 0, 0.2
)

# plot_relademand_cdf(
#     PREFIXES, LABELS, FIGDIR + CASE_NAME + "relademand.png", is_pf_schedule
# )

# plot_absvalue_multi_sys(
#     PREFIXES, LABELS, FIGDIR + CASE_NAME + "tp_first20pran", False, 0, 0.2
# )

# plot_absvalue_multi_sys(
#     PREFIXES, LABELS, FIGDIR + CASE_NAME + "tp_first20p", False, 0, 0.2, False
# )

# plot_absvalue_multi_sys(
#     PREFIXES, LABELS, FIGDIR + CASE_NAME + "ran_first20p", True, 0, 0.2
# )

# plot_improve_multi_sys(
#     0, PREFIXES, LABELS, FIGDIR + CASE_NAME + "ran_improve_first20p.png", True, 0, 0.2)

# plot_improve_multi_sys(
#     0, PREFIXES, LABELS, FIGDIR + CASE_NAME + "tp_improve_first20p.png", False, 0, 0.2)
