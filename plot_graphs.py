#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

n_samples = 20

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


def plot_scheme_compare(f1: str, f2: str, f3: str, ofname: str, is_ran: True):
    """
    Plot per-UE ran&data_rate comparison across multiple schemes
    """
    dfs = []
    if not os.path.exists(f2):
        return
    dfs.append(pd.read_csv(f2))
    dfs.append(pd.read_csv(f3))
    # dfs.append( pd.read_csv(f3) )
    # schemes = ["Origin", "NaiveHO", "SmartHO"]
    schemes = ["Origin", "NaiveHO"]
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
                x_array, data_sort, width=bar_width, label=schemes[i] + "-S" + str(sid)
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
    for sid in range(n_slices):
        ue_metrics = []
        slice_ues = df[df["slice"] == sid]
        for _, row in slice_ues.iterrows():
            metric = row["ran"] if is_ran else row["ran"] * row["channel"]
            ue_metrics.append(metric)
        ue_metrics = get_sublist(sorted(ue_metrics), lowb, upperb)
    return sorted(ue_metrics)


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
    print(sort_gain)
    print(len(sort_gain))
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
    sys1: int, sys2: int, sys3: int, ofname: str, is_ran: bool, lowb: float, upperb: float
):
    fig, ax = plt.subplots(figsize=(8, 6))
    scheme1_metrics = []
    scheme2_metrics = []
    scheme3_metrics = []
    for i in range(n_samples):
        if not os.path.exists(SDIR + PREFIXES[sys1] + str(i) + ".csv"):
            continue
        s1_metrics = get_metric_value(SDIR + PREFIXES[sys1] + str(i) + ".csv", lowb, upperb, is_ran)
        s2_metrics = get_metric_value(SDIR + PREFIXES[sys2] + str(i) + ".csv", lowb, upperb, is_ran)
        s3_metrics = get_metric_value(SDIR + PREFIXES[sys3] + str(i) + ".csv", lowb, upperb, is_ran)
        scheme1_metrics = scheme1_metrics + s1_metrics
        scheme2_metrics = scheme2_metrics + s2_metrics
        scheme3_metrics = scheme3_metrics + s3_metrics
    ax.ecdf(scheme1_metrics, label=LABELS[sys1])
    ax.ecdf(scheme2_metrics, label=LABELS[sys2])
    ax.ecdf(scheme3_metrics, label=LABELS[sys3])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlim(left=0)
    ax.legend()
    if is_ran:
        ax.set_xlabel("RAN allocation")
    else:
        ax.set_xlabel("Throughput(Mbps)")
    ax.set_ylabel("Probability")
    fig.savefig(ofname)


def plot_improve_multi_samples(
    sys1: int, sys2: int, sys3: int, ofname: str, is_ran: bool, lowb: float, upperb: float
):
    """
    Plot the CDF graph of the metric improvement from sys1 to sys2 and sys3
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    all_improve1 = []
    all_improve2 = []
    for i in range(n_samples):
        if not os.path.exists(SDIR + PREFIXES[sys1] + str(i) + ".csv"):
            continue
        improve1 = get_sameue_improve(
            SDIR + PREFIXES[sys1] + str(i) + ".csv",
            SDIR + PREFIXES[sys2] + str(i) + ".csv",
            lowb, upperb, is_ran
        )
        improve2 = get_sameue_improve(
            SDIR + PREFIXES[sys1] + str(i) + ".csv",
            SDIR + PREFIXES[sys3] + str(i) + ".csv",
            lowb, upperb, is_ran
        )
        all_improve1 = all_improve1 + improve1
        all_improve2 = all_improve2 + improve2
    ax.ecdf(sorted(all_improve1), label="Origin->NaiveHO")
    ax.ecdf(sorted(all_improve2), label="Origin->SmartHO")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    if sorted(all_improve1)[-1] > 4 or sorted(all_improve2)[-1] > 4:
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


SDIR = "./sample_data/"
# plot_perue_bw(SDIR+"naiveho5.csv", "datarate_naiveho5.png")
# plot_perue_ran(SDIR+"naiveho5.csv", "ran_naiveho5.png")
# plot_perue_bw(SDIR+"smartho5.csv", "datarate_smartho5.png")
# plot_perue_ran(SDIR+"smartho5.csv", "ran_smartho5.png")

# for i in range(10):
#     plot_scheme_compare(SDIR+"origin_c5_" + str(i) + ".csv",
#                        SDIR+"naiveho_c5_" + str(i) + ".csv",
#                        SDIR+"smartho_c5_" + str(i) + ".csv",
#                        SDIR+"compare_c5_ran" + str(i) + ".png", True)
#     plot_scheme_compare(SDIR+"origin_c5_" + str(i) + ".csv",
#                        SDIR+"naiveho_c5_" + str(i) + ".csv",
#                        SDIR+"smartho_c5_" + str(i) + ".csv",
#                        SDIR+"compare_c5_datarate" + str(i) + ".png", False)

LABELS=["Origin", "NaiveHO", "UnawareHO", "AwareHO"]
PREFIXES=["origin_", "naiveho_", "unawareho_", "awareho_"]

plot_absvalue_multi_samples(0, 1, 2, "tp_first20p.png", False, 0, 0.2)

plot_absvalue_multi_samples(0, 1, 2, "ran_first20p.png", True, 0, 0.2)

plot_improve_multi_samples(0, 1, 2, "ran_improve_first20p.png", True, 0, 0.2)

plot_improve_multi_samples(0, 1, 2, "tp_improve_first20p.png", False, 0, 0.2)


# for i in range(10):
#     plot_scheme_compare(SDIR+"origin_screw_" + str(i) + ".csv",
#                        SDIR+"naiveho_screw_" + str(i) + ".csv",
#                        SDIR+"smartho_screw_" + str(i) + ".csv",
#                        SDIR+"compare_screw_ran" + str(i) + ".png", True)
#     plot_scheme_compare(SDIR+"origin_screw_" + str(i) + ".csv",
#                        SDIR+"naiveho_screw_" + str(i) + ".csv",
#                        SDIR+"smartho_screw_" + str(i) + ".csv",
#                        SDIR+"compare_screw_datarate" + str(i) + ".png", False)
# plot_absvalue_multi_samples(
#     "origin_screw_", "naiveho_screw_", "unawareho_screw_", "tp_screw_first20p.png", False, 0, 0.2
# )
# plot_absvalue_multi_samples(
#     "origin_screw_", "naiveho_screw_", "unawareho_screw_", "ran_screw_first20p.png", True, 0, 0.2
# )
# plot_improve_multi_samples(
#     "origin_screw_", "naiveho_screw_", "unawareho_screw_", "ran_improve_screw_first20p.png", True, 0, 0.2
# )
# plot_improve_multi_samples(
#     "origin_screw_", "naiveho_screw_", "unawareho_screw_", "tp_improve_screw_first20p.png", False, 0, 0.2
# )
