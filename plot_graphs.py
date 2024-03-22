#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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


def plot_scheme_compare(fnames: list, ofname: str, is_ran: True):
    """
    Plot per-UE ran&data_rate comparison across multiple schemes
    """
    dfs = []
    if not os.path.exists(fnames[0]):
        return
    for fname in fnames:
        dfs.append(pd.read_csv(fname))
    schemes = ["NaiveHO", "UnawareHO"]
    # schemes = ["Origin", "NaiveHO", "UnawareHO"]
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
            x_array = np.arange(ue_begin, ue_begin + len(data_sort)) + bar_width * (i - 0.5)
            ax.bar(x_array, data_sort,
                   width=bar_width,
                   label=schemes[i] + "-S" + str(sid))
            ue_begin += len(data_sort)
    ax.set_xlabel("UE ID")
    if is_ran:
        ax.set_ylabel("RAN percentage")
    else:
        ax.set_ylabel("Throughput(Mbps)")
    ax.set_ylim((0, 20))
    ax.legend()
    plt.tight_layout()
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
    df = pd.read_csv(fname)
    n_slices = df["slice"].max() + 1
    scheme1_array = []
    for sid in range(n_slices):
        s1_ues = sorted(list(df[df["slice"] == sid]["ran"]))
        scheme1_array.append(s1_ues[-1] / s1_ues[0])
    return scheme1_array


def get_metric_value(fname: str, lowb: float, upperb: float, is_ran: True):
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


def get_pfmetric(f1, f2):
    """
    it returns the per-slice pf-metric of f1 and f2
    """
    dfs = []
    dfs.append(pd.read_csv(f1))
    dfs.append(pd.read_csv(f2))
    n_slices = dfs[0]["slice"].max() + 1
    pf_metrics_scheme = []
    for i, df in enumerate(dfs):
        pf_metrics = []
        for sid in range(n_slices):
            slice_ues = df[df["slice"] == sid]
            cqi = 72
            # cqi = 50
            pf_metric = 0
            for _, row in slice_ues.iterrows():
                datarate = float(row["ran"]) * cqi
                pf_metric += np.log(datarate)
            pf_metrics.append(pf_metric)
        pf_metrics_scheme.append(pf_metrics)
    return pf_metrics_scheme


def plot_pfmetric_gain(ofname):
    n_samples = 30
    SDIR = "./sample_result/"
    gain_array = []
    naive_pf_array = []
    smart_pf_array = []
    for i in range(n_samples):
        metrics = get_pfmetric(
            SDIR + "naiveho_c4_" + str(i) + ".csv",
            SDIR + "smartho_c4_" + str(i) + ".csv",
        )
        for j in range(len(metrics[0])):
            if metrics[0][j] < 0:
                continue
            gain_array.append((metrics[1][j] / metrics[0][j] - 1) * 100)
            naive_pf_array.append(metrics[0][j])
            smart_pf_array.append(metrics[1][j])
    sort_gain = sorted(gain_array)
    cdf = np.linspace(0, 1, len(sort_gain))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(sort_gain, cdf)
    axs[0].set_xlabel("PFMetric_gain(%)")
    axs[0].set_ylabel("Percentage")
    axs[0].set_title("PF-Metric gain of SmartHO compared with NaiveHO")
    axs[1].plot(sorted(naive_pf_array), cdf, label="NaiveHO")
    axs[1].plot(sorted(smart_pf_array), cdf, label="SmartHO")
    axs[1].set_xlabel("PFMetric")
    axs[1].set_ylabel("Percentage")
    axs[1].set_title("Absolute PF metrics")
    axs[1].legend()
    fig.savefig(ofname)


def plot_absvalue_multi_samples(
    prefixes: list, labels: list, ofname: str, is_ran: bool, lowb: float, upperb: float
):
    line_styles = ["solid", "dotted", "dashed", "dashdot"]
    fig, ax = plt.subplots(figsize=(8, 6))
    scheme_metrics = [[] for i in range(len(prefixes))]
    for i in range(n_samples):
        if not os.path.exists(SDIR + prefixes[0] + str(i) + ".csv"):
            continue
        for j, prefix in enumerate(prefixes):
            metrics = get_metric_value(SDIR + prefix + str(i) + ".csv", lowb, upperb, is_ran)
            scheme_metrics[j] = scheme_metrics[j] + metrics
    for j, label in enumerate(labels):
        ax.ecdf(scheme_metrics[j], label=label, linestyle=line_styles[j])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    if is_ran:
        ax.set_xlabel("RAN allocation")
    else:
        ax.set_xlabel("Throughput(Mbps)")
    ax.set_ylabel("Probability")
    fig.savefig(ofname)


def plot_improve_multi_samples(
    basic_id: int, prefixes: list, labels: list, ofname: str, is_ran: bool, lowb: float, upperb: float
):
    """
    Plot the CDF graph of the metric improvement from sys1 to sys2 and sys3
    """
    line_styles = [ "dashdot", "solid", "dotted", "dashed"]
    fig, ax = plt.subplots(figsize=(8, 6))
    all_improves = [[] for i in range(len(prefixes))]
    for i in range(n_samples):
        if not os.path.exists(SDIR + prefixes[basic_id] + str(i) + ".csv"):
            continue
        for j, prefix in enumerate(prefixes):
            improve = get_sameue_improve(
                  SDIR + PREFIXES[basic_id] + str(i) + ".csv",
                  SDIR + prefix + str(i) + ".csv",
                  lowb, upperb, is_ran)
            all_improves[j] = all_improves[j] + improve
    max_value = 0
    for j, label in enumerate(labels):
        if j != basic_id:
          all_improves[j] = sorted(all_improves[j])
          ax.ecdf(all_improves[j], label=labels[basic_id] + "->" + label, linestyle=line_styles[j])
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
        np.arange(len(scheme1_ratio)) - width, scheme1_ratio, width=width, label=LABELS[sys1]
    )
    ax.bar(
        np.arange(len(scheme2_ratio)), scheme2_ratio, width=width, label=LABELS[sys2]
    )
    ax.bar(
        np.arange(len(scheme3_ratio)) + width, scheme3_ratio, width=width, label=LABELS[sys3]
    )
    ax.legend()
    ax.set_xlabel("Slice-ID")
    ax.set_ylabel("Max-Min Ratio")
    fig.savefig(ofname)

is_pf_schedule = True
if is_pf_schedule:
    SDIR="./sample_data/"
else:
    SDIR="./sample_data_df/"

n_samples = 20
LABELS=["Origin", "NaiveHO", "UnawareHO", "AwareHO"]
case_name="basic_"
PREFIXES = ["origin_", "naiveho_", "unawareho_","awareho_"]
PREFIXES = [prefix + case_name for prefix in PREFIXES]

# for i in range(10):
#     plot_scheme_compare(
#         [
#           SDIR+PREFIXES[1] + str(i) + ".csv",
#           SDIR+PREFIXES[2] + str(i) + ".csv"],
#         "./figures/" + case_name + "compare_ran" + str(i) + ".png",
#         True)
#     plot_scheme_compare(
#         [
#          SDIR+PREFIXES[1] + str(i) + ".csv",
#          SDIR+PREFIXES[2] + str(i) + ".csv"],
#         "./figures/" + case_name + "compare_datarate" + str(i) + ".png",
#         False)

plot_absvalue_multi_samples(
    PREFIXES, LABELS, "./figures/" + case_name + "tp_first20p.png", False, 0, 0.2)

plot_absvalue_multi_samples(
    PREFIXES, LABELS, "./figures/" + case_name + "ran_first20p.png", True, 0, 0.2)

plot_improve_multi_samples(
    0, PREFIXES, LABELS, "./figures/" + case_name + "ran_improve_first20p.png", True, 0, 0.2)

plot_improve_multi_samples(
    0, PREFIXES, LABELS, "./figures/" + case_name + "tp_improve_first20p.png", False, 0, 0.2)

# LABELS=["Origin", "NaiveHO", "UnawareHO", "AwareHO"]
# PREFIXES=["origin_macro_", "naiveho_macro_", "unawareho_macro_", "awareho_macro_"]

# plot_absvalue_multi_samples(PREFIXES, LABELS, "tp_macro_first20p.png", False, 0, 0.2)

# plot_absvalue_multi_samples(PREFIXES, LABELS, "ran_macro_first20p.png", True, 0, 0.2)

# plot_improve_multi_samples(0, PREFIXES, LABELS, "ran_improve_macro_first20p.png", True, 0, 0.2)

# plot_improve_multi_samples(0, PREFIXES, LABELS, "tp_improve_macro_first20p.png", False, 0, 0.2)
