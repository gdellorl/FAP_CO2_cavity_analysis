#!/usr/bin/env python3
"""
cavity_analysis.py
=============================================================
Step 2 of the CO2 cavity analysis (run cavity_precompute.py first).

Reads the residue-contact cache and the same input file used by
cavity_precompute.py, then:
  1. detects candidate cavity visits in each trajectory (consensus reference
     set + Jaccard-based persistence tracking with recrossing tolerance);
  2. clusters the visits of all trajectories into cavities (average-linkage
     hierarchical clustering on d = 1 - [w*J + (1-w)*O]);
  3. computes per-cavity populations and residence times;
  4. plots the cavity-occupancy timeline (one panel per trajectory) and its
     legend as two separate figures;
  5. clusters the CO2 exit sites computed by cavity_precompute.py (residues
     contacted by CO2 just before its first exit) into exit regions.

Usage:
    python3 cavity_analysis.py input.txt

Main outputs (written to 'output_dir'):
    cavity_detection_merged.dat     raw cavity visits
    cavity_groups_merged.dat        final cavities (ordered by population) and their visits
    cavity_residues_readable.txt    consensus residues of each cavity
    cavity_residence_summary.dat    populations and residence-time statistics
    cavity_events/                  list of visits of each cavity
    cavity_timeline_panels.png      timeline figure (no legend)
    cavity_timeline_legend.png      legend of the timeline figure
    exit_sites_clusters.txt         exit regions, their residues and the similarity matrix

Requires: numpy, scipy, matplotlib
=============================================================
"""

import os
import re
import sys
import shlex
import pickle
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ======================================================
# --- Input file parsing (shared format with cavity_precompute.py) ---
# ======================================================
DEFAULT_COLORS = [
    (0.12, 0.47, 0.71),   # Cavity 1
    (0.68, 0.78, 0.91),   # Cavity 2
    (1.00, 0.50, 0.05),   # Cavity 3
    (1.00, 0.73, 0.47),   # Cavity 4
    (0.17, 0.63, 0.17),   # Cavity 5
    (0.60, 0.87, 0.54),   # Cavity 6
    (0.84, 0.15, 0.16),   # Cavity 7
]

# key: (type, default)
PARAMETERS = {
    # --- files ---
    "cache_file": ("str", "residues_cache.pkl"),
    "output_dir": ("str", "."),
    # --- contacts (cavity_precompute.py) ---
    "contact_cutoff_nm": ("float", 0.4),
    "target_selection": ("str", "(protein or resname FAD or resname C15) and not (name H* or type H)"),
    "ligand_selection": ("str", "resname CO2"),
    # --- first exit of the ligand from the protein (cavity_precompute.py) ---
    "exit_selection": ("str", "protein"),
    "exit_threshold_nm": ("float", 0.36),
    "exit_contact_cutoff_nm": ("float", 0.32),
    "exit_window_ps": ("float", 100.0),
    # --- cavity detection ---
    "dt_ps": ("float", 10.0),
    "reference_window_ps": ("float", 50.0),
    "jaccard_threshold": ("float", 0.4),
    "recrossing_tolerance_ps": ("float", 20.0),
    "consensus_fraction": ("float", 0.5),
    "min_visit_ps": ("float", 100.0),
    "min_residues_cavity": ("int", 3),
    # --- clustering ---
    "similarity_threshold": ("float", 0.36),
    "jaccard_weight": ("float", 0.5),
    # --- exit-region clustering ---
    "exit_similarity_threshold": ("float", 0.36),
    "exit_jaccard_weight": ("float", 0.5),
    # --- figures ---
    "n_explicit_cavities": ("int", 6),
    "cavity_colors": ("colors", DEFAULT_COLORS),
    "other_color": ("color", (0.75, 0.75, 0.75, 1.0)),
    "gap_color": ("color", "#F5E6A3"),
    "outside_color": ("color", "white"),
    "outside_edge_color": ("color", "black"),
    "panel_order": ("str", "name"),
    "figsize": ("floats", (8.0, 16.2)),
    "vspace": ("float", 0.03),
    "font_label": ("float", 20),
    "font_tick": ("float", 16),
    "legend_title": ("str", "Cavity legend"),
    "font_legend": ("float", 18),
    "font_legend_title": ("float", 20),
    "dpi": ("int", 300),
    "timeline_file": ("str", "cavity_timeline_panels.png"),
    "legend_file": ("str", "cavity_timeline_legend.png"),
}


def _parse_color(token):
    """'#1f77b4', 'tab:blue', 'white' or 'r,g,b[,a]' with values in [0,1]."""
    if "," in token:
        return tuple(float(x) for x in token.split(","))
    return token


def _convert(key, kind, text):
    try:
        if kind == "float":
            return float(text)
        if kind == "int":
            return int(text)
        if kind == "str":
            return text
        if kind == "floats":
            return tuple(float(x) for x in text.split())
        if kind == "color":
            return _parse_color(text)
        if kind == "colors":
            return [_parse_color(t) for t in text.split()]
    except ValueError:
        pass
    raise ValueError(f"Invalid value for '{key}': {text!r} (expected {kind})")


def _strip_comment(line):
    """Remove full-line comments and inline comments of the form ' # ...'.
    A '#' not preceded by whitespace (e.g. a hex color '#F5E6A3') is kept."""
    if line.lstrip().startswith("#"):
        return ""
    return re.split(r"\s#(?:\s|$)", line, maxsplit=1)[0].strip()


def read_input(path):
    """Return (params, trajectories), with trajectories = [(name, topology, trajectory), ...].
    Relative paths are interpreted relative to the input file location."""
    params = {key: default for key, (_, default) in PARAMETERS.items()}
    trajectories = []
    base_dir = os.path.dirname(os.path.abspath(path))
    section = None
    with open(path) as f:
        for n_line, raw in enumerate(f, 1):
            line = _strip_comment(raw)
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                if section != "trajectories":
                    raise ValueError(f"{path}:{n_line}: unknown section [{section}]")
                continue
            if section == "trajectories":
                tokens = shlex.split(line)
                if len(tokens) != 3:
                    raise ValueError(f"{path}:{n_line}: expected 'name topology trajectory', got {line!r}")
                name, top, trj = tokens
                top = top if os.path.isabs(top) else os.path.join(base_dir, top)
                trj = trj if os.path.isabs(trj) else os.path.join(base_dir, trj)
                trajectories.append((name, top, trj))
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{n_line}: expected 'key = value', got {line!r}")
            key, value = (s.strip() for s in line.split("=", 1))
            if key not in PARAMETERS:
                raise ValueError(f"{path}:{n_line}: unknown parameter '{key}'")
            params[key] = _convert(key, PARAMETERS[key][0], value)
    names = [t[0] for t in trajectories]
    if len(set(names)) != len(names):
        raise ValueError("Trajectory names in the input file must be unique")
    if not trajectories:
        raise ValueError("No trajectories listed in the [trajectories] section")
    return params, trajectories


def ps_to_frames(key, value_ps, dt_ps):
    """Convert a time parameter (ps) into an integer number of frames."""
    n = int(round(value_ps / dt_ps))
    if abs(n * dt_ps - value_ps) > 1e-6:
        raise ValueError(f"'{key}' = {value_ps} ps is not a multiple of dt_ps = {dt_ps} ps")
    return n


def natural_key(text):
    """Sort 'MD2' before 'MD10'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


# ======================================================
# --- Set similarity ---
# ======================================================
def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def overlap_coefficient(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def combined_similarity(a, b, jaccard_weight):
    return jaccard_weight * jaccard(a, b) + (1 - jaccard_weight) * overlap_coefficient(a, b)


# ======================================================
# --- Exit regions ---
# ======================================================
def cluster_exit_sites(exit_sets, P, out_path):
    """Average-linkage clustering of the exit sites (same similarity measure used for
    cavities). Writes the exit regions, their residues and the pairwise similarity
    matrix to 'out_path', and returns the number of exit regions."""
    names = sorted(exit_sets, key=natural_key)
    n = len(names)
    w = P["exit_jaccard_weight"]
    s_exit = P["exit_similarity_threshold"]

    sim = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = combined_similarity(exit_sets[names[i]], exit_sets[names[j]], w)

    if n == 1:
        labels = np.array([1])
    else:
        Z = linkage(squareform(1 - sim, checks=False), method="average")
        labels = fcluster(Z, t=1 - s_exit, criterion="distance")

    members = {}
    for name, lab in zip(names, labels):
        members.setdefault(lab, []).append(name)
    # exit regions ordered by number of trajectories, then by trajectory name
    regions = sorted(members.values(), key=lambda m: (-len(m), natural_key(m[0])))

    with open(out_path, "w") as f:
        f.write(f"# Exit regions (similarity threshold = {s_exit}, jaccard_weight = {w})\n")
        f.write("# Two exit sites belong to the same region if their average similarity\n"
                "# (average linkage) is >= the similarity threshold.\n\n")
        for k, region in enumerate(regions, 1):
            union = sorted(set().union(*(exit_sets[nm] for nm in region)))
            f.write(f"Exit region {k}: {', '.join(region)}\n")
            f.write(f"  Residues ({len(union)}): {' '.join(map(str, union))}\n\n")
        f.write("# Exit residues of each trajectory\n")
        width = max(len(nm) for nm in names)
        for nm in names:
            f.write(f"{nm:>{width}}: {' '.join(map(str, sorted(exit_sets[nm])))}\n")
        f.write("\n# Pairwise similarity (0 = no residue in common, 1 = identical residue sets)\n")
        col = width + 1
        f.write(" " * col + "".join(f"{nm:>{col}}" for nm in names) + "\n")
        for nm, row in zip(names, sim):
            f.write(f"{nm:>{col}}" + "".join(f"{v:>{col}.2f}" for v in row) + "\n")

    for k, region in enumerate(regions, 1):
        print(f"  Exit region {k}: {', '.join(region)}")
    return len(regions)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {os.path.basename(sys.argv[0])} input.txt")

    P, trajectories_in = read_input(sys.argv[1])
    out_dir = P["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def out(*parts):
        return os.path.join(out_dir, *parts)

    dt_ps = P["dt_ps"]
    dt_ns = dt_ps / 1000.0
    n_frame_reference = ps_to_frames("reference_window_ps", P["reference_window_ps"], dt_ps)
    recrossing_tolerance = ps_to_frames("recrossing_tolerance_ps", P["recrossing_tolerance_ps"], dt_ps)
    n_frame_min = ps_to_frames("min_visit_ps", P["min_visit_ps"], dt_ps)
    jaccard_threshold = P["jaccard_threshold"]
    consensus_fraction = P["consensus_fraction"]
    min_residues_cavity = P["min_residues_cavity"]
    S = P["similarity_threshold"]
    jaccard_weight = P["jaccard_weight"]

    # ======================================================
    # --- Load the contact cache ---
    # ======================================================
    with open(out(P["cache_file"]), "rb") as f:
        cache = pickle.load(f)

    residues_per_frame = cache["residues_per_frame"]
    frame_labels = cache["frame_labels"]
    boundaries = cache["boundaries"]
    boundary_end = cache["boundary_end"]
    trajectory_labels_order = cache["trajectory_labels_order"]
    n_tot = len(residues_per_frame)

    cache_cutoff = cache.get("contact_cutoff_nm", cache.get("soglia_vicinanza"))  # older caches
    print(f"Cache loaded: {n_tot} frames, contact cutoff = {cache_cutoff} nm")
    if [t[0] for t in trajectories_in] != list(trajectory_labels_order):
        print("[warning] trajectory names/order in the input file differ from those in the cache; "
              "the cache is used. Re-run cavity_precompute.py if this is not intended.")
    if cache_cutoff is not None and abs(cache_cutoff - P["contact_cutoff_nm"]) > 1e-9:
        print(f"[warning] contact_cutoff_nm in the input file ({P['contact_cutoff_nm']}) differs from "
              f"the cache ({cache_cutoff}); the cache is used.")

    # time with at least one CO2-residue contact (confirmed cavities + gaps)
    total_contact_frames = sum(1 for s in residues_per_frame if len(s) > 0)
    print(f"Frames with at least one contact: {total_contact_frames} "
          f"({100 * total_contact_frames / n_tot:.1f}% of the total simulated time)")

    traj_n_frames = {}
    for i, label in enumerate(trajectory_labels_order):
        start = boundaries[i]
        traj_n_frames[label] = boundary_end[start] - start + 1
    traj_index = {label: i for i, label in enumerate(trajectory_labels_order)}

    def consensus_reference(frame_indices, fraction):
        """Residues present in at least 'fraction' of the given frames (majority vote)."""
        counter = Counter()
        for idx in frame_indices:
            counter.update(residues_per_frame[idx])
        threshold = fraction * len(frame_indices)
        return {resid for resid, count in counter.items() if count >= threshold}

    def end_of_current_traj(t):
        current_start = max(b for b in boundaries if b <= t)
        return boundary_end[current_start]

    # ======================================================
    # --- STEP 1: detection of cavity visits ---
    # ======================================================
    cavities = []
    t0 = 0
    while t0 < n_tot:
        traj_end = end_of_current_traj(t0)

        # reference window would cross the end of the trajectory: jump to the next one
        if t0 + n_frame_reference - 1 > traj_end:
            next_boundaries = [b for b in boundaries if b > t0]
            if not next_boundaries:
                break
            t0 = min(next_boundaries)
            continue

        ref_window = list(range(t0, t0 + n_frame_reference))
        if all(len(residues_per_frame[f]) == 0 for f in ref_window):
            t0 += 1
            continue

        cavity_ref = consensus_reference(ref_window, consensus_fraction)
        if len(cavity_ref) < min_residues_cavity:
            t0 += 1
            continue

        # follow the reference set forward until J stays below threshold for
        # more than 'recrossing_tolerance' consecutive frames
        bad_count = 0
        t_exit = None
        g = t0 + 1
        while g <= traj_end:
            if jaccard(residues_per_frame[g], cavity_ref) < jaccard_threshold:
                bad_count += 1
                if bad_count > recrossing_tolerance:
                    t_exit = g - bad_count + 1
                    break
            else:
                bad_count = 0
            g += 1
        if t_exit is None:
            t_exit = traj_end + 1

        duration = t_exit - t0
        if duration < n_frame_min:
            t0 += 1
            continue

        # final residue set of the visit: consensus over all its frames
        cavity_final = consensus_reference(list(range(t0, t_exit)), consensus_fraction)
        if len(cavity_final) < min_residues_cavity:
            cavity_final = cavity_ref
        if len(cavity_final) < min_residues_cavity:
            t0 = t_exit
            continue

        label_start, local_start = frame_labels[t0]
        _, local_end_incl = frame_labels[t_exit - 1]
        cavities.append({
            "label": label_start,
            "t_start_local": local_start,
            "t_exit_local": local_end_incl + 1,
            "duration_frames": duration,
            "residues": sorted(cavity_final),
        })
        t0 = t_exit

    n_raw = len(cavities)
    print(f"Raw cavity visits detected: {n_raw}")
    if n_raw < 2:
        sys.exit("Fewer than two cavity visits detected: nothing to cluster. "
                 "Check the input trajectories and detection parameters.")

    with open(out("cavity_detection_merged.dat"), "w") as fout:
        fout.write("# label t_start_local t_exit_local duration_frames residues...\n")
        for c in cavities:
            fout.write(f"{c['label']} {c['t_start_local']} {c['t_exit_local']} {c['duration_frames']} "
                       + " ".join(str(r) for r in c["residues"]) + "\n")

    # ======================================================
    # --- STEP 2: clustering of visits into cavities (average linkage) ---
    # ======================================================
    raw_sets = [set(c["residues"]) for c in cavities]
    n = len(raw_sets)

    def combined_distance(a, b):
        sim = jaccard_weight * jaccard(a, b) + (1 - jaccard_weight) * overlap_coefficient(a, b)
        return 1 - sim

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = combined_distance(raw_sets[i], raw_sets[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    Z = linkage(squareform(dist_matrix, checks=False), method="average")
    cluster_labels = fcluster(Z, t=1 - S, criterion="distance")
    n_clusters = len(set(cluster_labels))
    print(f"Similarity threshold S = {S} -> {n_clusters} cavities")

    groups_dict = {}
    for idx, cl in enumerate(cluster_labels):
        groups_dict.setdefault(cl, []).append(idx)

    def total_frames_of_group(members):
        return sum(cavities[idx]["duration_frames"] for idx in members)

    # cavities are numbered by population (total residence time): Cavity 1 = most populated
    groups_idx = sorted(groups_dict.values(), key=lambda g: -total_frames_of_group(g))

    def pooled_consensus(members):
        frames = []
        for idx in members:
            c = cavities[idx]
            offset = boundaries[traj_index[c["label"]]]
            frames.extend(range(offset + c["t_start_local"], offset + c["t_exit_local"]))
        return consensus_reference(frames, consensus_fraction)

    group_residues = [sorted(pooled_consensus(members)) for members in groups_idx]

    with open(out("cavity_groups_merged.dat"), "w") as fout:
        fout.write(f"# S={S} -- group_id n_visits consensus_residues | (label:t_start_local-t_exit_local,...)\n")
        for gid, members in enumerate(groups_idx):
            visits = ",".join(f"{cavities[i]['label']}:{cavities[i]['t_start_local']}-{cavities[i]['t_exit_local']}"
                              for i in members)
            fout.write(f"{gid} {len(members)} {' '.join(map(str, group_residues[gid]))} | {visits}\n")

    # ======================================================
    # --- STEP 3: populations and residence times ---
    # ======================================================
    events_dir = out("cavity_events")
    os.makedirs(events_dir, exist_ok=True)

    total_all_cavities_ps = sum(total_frames_of_group(m) for m in groups_idx) * dt_ps
    summary_rows = []

    for gid, members in enumerate(groups_idx):
        cav_num = gid + 1
        events = sorted(
            (cavities[idx] for idx in members),
            key=lambda c: (traj_index[c["label"]], c["t_start_local"]),
        )
        durations_ps = np.array([c["duration_frames"] * dt_ps for c in events])
        n_events = len(events)

        # the second column (md_name) repeats the trajectory name, for compatibility
        # with scripts reading the earlier format of these files
        with open(os.path.join(events_dir, f"cavity_{cav_num:03d}_events.dat"), "w") as fdat:
            fdat.write(f"{'#trajectory_label':<20}{'md_name':>10}{'t_start_frame':>16}{'t_exit_frame':>16}"
                       f"{'t_start_ps':>16}{'t_exit_ps':>16}{'duration_frames':>18}{'duration_ps':>16}\n")
            for c in events:
                fdat.write(f"{c['label']:<20}{c['label']:>10}"
                           f"{c['t_start_local']:>16d}{c['t_exit_local']:>16d}"
                           f"{c['t_start_local'] * dt_ps:>16.1f}{c['t_exit_local'] * dt_ps:>16.1f}"
                           f"{c['duration_frames']:>18d}{c['duration_frames'] * dt_ps:>16.1f}\n")

        total_res_ps = float(durations_ps.sum())
        std = float(durations_ps.std(ddof=1)) if n_events > 1 else 0.0
        summary_rows.append({
            "cavity": cav_num,
            "n_events": n_events,
            "total_res_ps": total_res_ps,
            "pct_cavities": 100 * total_res_ps / total_all_cavities_ps if total_all_cavities_ps > 0 else 0.0,
            "mean": float(durations_ps.mean()),
            "sem": std / np.sqrt(n_events) if n_events > 1 else 0.0,
            "median": float(np.median(durations_ps)),
            "std": std,
        })

    # pct_cavities_time_% : population, i.e. residence time of the cavity relative to the
    #                       total time spent in all detected cavities
    # std/sem             : sample standard deviation (ddof=1) and standard error of the mean
    with open(out("cavity_residence_summary.dat"), "w") as fdat:
        fdat.write(f"{'#cavity':<10}{'n_events':>12}{'total_res_ps':>18}"
                   f"{'pct_cavities_time_%':>22}"
                   f"{'mean_dur_ps':>16}{'sem_dur_ps':>14}{'median_dur_ps':>16}{'std_dur_ps':>14}\n")
        for r in summary_rows:
            fdat.write(f"{r['cavity']:<10d}{r['n_events']:>12d}{r['total_res_ps']:>18.1f}"
                       f"{r['pct_cavities']:>22.2f}"
                       f"{r['mean']:>16.1f}{r['sem']:>14.1f}{r['median']:>16.1f}{r['std']:>14.1f}\n")

    # readable summary of the residues lining each cavity
    tot_frames_assigned = sum(c["duration_frames"] for c in cavities)
    perc_assigned = 100 * tot_frames_assigned / n_tot
    with open(out("cavity_residues_readable.txt"), "w") as fout:
        fout.write("CAVITY SUMMARY - consensus residues of each cavity\n")
        fout.write("=" * 60 + "\n\n")
        for gid, members in enumerate(groups_idx):
            tot_frames = total_frames_of_group(members)
            fout.write(f"Cavity {gid + 1}  (gid={gid}, {len(members)} visits, {tot_frames} frames, "
                       f"{tot_frames * dt_ps:.0f} ps total residence time)\n")
            fout.write(f"  Residues ({len(group_residues[gid])}): "
                       + ", ".join(str(r) for r in group_residues[gid]) + "\n\n")
        fout.write("=" * 60 + "\n")
        fout.write(f"Total frames: {n_tot}\n")
        fout.write(f"Frames assigned to a cavity: {tot_frames_assigned} ({perc_assigned:.1f}%)\n")
        fout.write(f"Unassigned frames (gap + outside protein): {n_tot - tot_frames_assigned} "
                   f"({100 - perc_assigned:.1f}%)\n")

    print(f"Frames assigned to a cavity: {tot_frames_assigned} ({perc_assigned:.1f}%)")
    for r in summary_rows[:P["n_explicit_cavities"]]:
        print(f"  Cavity {r['cavity']}: {r['pct_cavities']:.2f}% of cavity occupancy, {r['n_events']} visits, "
              f"mean residence time {r['mean'] / 1000:.2f} +/- {r['sem'] / 1000:.2f} ns (SEM)")

    # ======================================================
    # --- STEP 4: timeline figure and legend (two separate files) ---
    # ======================================================
    n_groups = len(groups_idx)
    n_explicit = min(P["n_explicit_cavities"], n_groups)
    explicit_colors = [mcolors.to_rgba(c) for c in P["cavity_colors"][:n_explicit]]
    tab20 = plt.colormaps["tab20"]
    for i in range(len(explicit_colors), n_explicit):   # fallback if too few colors are given
        explicit_colors.append(tab20(i / 20))
    other_color = mcolors.to_rgba(P["other_color"])

    # colormap index: 0 = outside protein, 1 = gap (contact but no cavity), 2.. = cavities
    colors_list = ([P["outside_color"], P["gap_color"]] + explicit_colors
                   + [other_color] * (n_groups - n_explicit))
    cmap = mcolors.ListedColormap(colors_list)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(colors_list) + 0.5, 1), cmap.N)

    frame_group = {lb: np.full(traj_n_frames[lb], -1, dtype=int) for lb in trajectory_labels_order}
    has_contact = {lb: np.zeros(traj_n_frames[lb], dtype=bool) for lb in trajectory_labels_order}
    for global_idx in range(n_tot):
        label, local_frame = frame_labels[global_idx]
        if residues_per_frame[global_idx]:
            has_contact[label][local_frame] = True
    for gid, members in enumerate(groups_idx):
        for idx in members:
            c = cavities[idx]
            frame_group[c["label"]][c["t_start_local"]:c["t_exit_local"]] = gid

    def encode_for_plot(label):
        values = np.zeros(traj_n_frames[label], dtype=int)
        values[has_contact[label]] = 1
        assigned = frame_group[label] >= 0
        values[assigned] = frame_group[label][assigned] + 2
        return values

    if P["panel_order"] == "name":
        panel_order = sorted(trajectory_labels_order, key=natural_key)
    elif P["panel_order"] == "input":
        panel_order = list(trajectory_labels_order)
    else:
        sys.exit("panel_order must be 'name' or 'input'")

    n_traj = len(panel_order)
    t_max_global = max(traj_n_frames[lb] for lb in panel_order) * dt_ns

    fig = plt.figure(figsize=P["figsize"], constrained_layout=True)
    gs = fig.add_gridspec(n_traj, 1, hspace=P["vspace"])
    axes = []
    for i, label in enumerate(panel_order):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        values = encode_for_plot(label)
        time_ns = np.arange(len(values)) * dt_ns
        ax.imshow(values.reshape(1, -1), aspect="auto", cmap=cmap, norm=norm,
                  extent=[time_ns[0], time_ns[-1], 0, 1])
        ax.set_xlim(0, t_max_global)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=P["font_label"])
        ax.tick_params(axis="x", labelsize=P["font_tick"])
        if i < n_traj - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xlabel("Time (ns)", fontsize=P["font_label"])
    axes[-1].ticklabel_format(style="plain", axis="x", useOffset=False)
    fig.savefig(out(P["timeline_file"]), dpi=P["dpi"])
    plt.close(fig)

    handles = [mpatches.Patch(color=explicit_colors[i], label=f"Cavity {i + 1}") for i in range(n_explicit)]
    if n_groups > n_explicit:
        handles.append(mpatches.Patch(color=other_color, label="Other cavities"))
    handles.append(mpatches.Patch(color=P["gap_color"], label="Gap"))
    handles.append(mpatches.Patch(facecolor=P["outside_color"], edgecolor=P["outside_edge_color"],
                                  label="Outside protein"))
    title = None if P["legend_title"].lower() == "none" else P["legend_title"]
    fig_leg = plt.figure(figsize=(3.5, 0.5 * len(handles) + 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=1, fontsize=P["font_legend"], frameon=False,
                  title=title, title_fontsize=P["font_legend_title"], handlelength=1.5, labelspacing=0.9)
    fig_leg.savefig(out(P["legend_file"]), dpi=P["dpi"], bbox_inches="tight")
    plt.close(fig_leg)

    # ======================================================
    # --- STEP 5: clustering of exit sites into exit regions ---
    # ======================================================
    exit_summary = ""
    exit_sites = cache.get("exit_sites", {})
    if exit_sites:
        exit_sets = {name: set(e["residues"]) for name, e in exit_sites.items()}
        n_regions = cluster_exit_sites(exit_sets, P, out("exit_sites_clusters.txt"))
        exit_summary = f" {len(exit_sets)} exit sites -> {n_regions} exit regions."
    else:
        print("No exit sites in the cache (no exit detected, or cache written by an older "
              "version of cavity_precompute.py): exit-region clustering skipped.")

    print(f"\nSummary: {n_raw} raw visits -> {n_clusters} cavities (S={S}).{exit_summary} "
          f"Outputs written to '{os.path.abspath(out_dir)}'")


if __name__ == "__main__":
    main()
