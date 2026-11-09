#!/usr/bin/env python3
"""
cavity_precompute.py
=============================================================
Step 1 of the CO2 cavity analysis.

For every frame of every trajectory listed in the input file, this script
  - computes the set of residues in contact with the ligand (CO2): a residue
    is in contact if at least one of its heavy atoms lies within
    'contact_cutoff_nm' of any ligand atom;
  - computes the ligand-protein minimum distance and detects the first exit
    of the ligand (first frame with minimum distance > 'exit_threshold_nm');
  - lists the residues in contact with the ligand in the 'exit_window_ps'
    preceding the exit (exit sites).
The results are stored in a pickle cache read by cavity_analysis.py, so that
detection/clustering parameters can be explored without re-reading the
trajectories.

Usage:
    python3 cavity_precompute.py input.txt

Input file format (see input.txt for a complete, commented example):
    key = value                      # parameters (defaults used if omitted)
    [trajectories]
    name   topology.pdb   trajectory.xtc

Requires: MDAnalysis, numpy
=============================================================
"""

import os
import re
import sys
import shlex
import pickle

# ======================================================
# --- Input file parsing (shared format with cavity_analysis.py) ---
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

# key: (type, default). Defaults are the values used in the published analysis.
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
    # --- cavity detection (cavity_analysis.py) ---
    "dt_ps": ("float", 10.0),
    "reference_window_ps": ("float", 50.0),
    "jaccard_threshold": ("float", 0.4),
    "recrossing_tolerance_ps": ("float", 20.0),
    "consensus_fraction": ("float", 0.5),
    "min_visit_ps": ("float", 100.0),
    "min_residues_cavity": ("int", 3),
    # --- clustering (cavity_analysis.py) ---
    "similarity_threshold": ("float", 0.36),
    "jaccard_weight": ("float", 0.5),
    # --- exit-region clustering (cavity_analysis.py) ---
    "exit_similarity_threshold": ("float", 0.36),
    "exit_jaccard_weight": ("float", 0.5),
    # --- figures (cavity_analysis.py) ---
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


# ======================================================
# --- Contacts, minimum distance and first exit ---
# ======================================================
def ps_to_frames(key, value_ps, dt_ps):
    """Convert a time parameter (ps) into an integer number of frames."""
    n = int(round(value_ps / dt_ps))
    if abs(n * dt_ps - value_ps) > 1e-6:
        raise ValueError(f"'{key}' = {value_ps} ps is not a multiple of dt_ps = {dt_ps} ps")
    return n


def residues_within(atoms, ligand, cutoff_nm, box):
    """Resids of 'atoms' having at least one atom within cutoff_nm of any ligand atom."""
    from MDAnalysis.analysis.distances import distance_array
    dists = distance_array(atoms.positions, ligand.positions, box=box) / 10.0   # Angstrom -> nm
    return set(atoms[(dists <= cutoff_nm).any(axis=1)].resids.tolist())


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {os.path.basename(sys.argv[0])} input.txt")

    params, trajectories = read_input(sys.argv[1])

    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array

    cutoff = params["contact_cutoff_nm"]
    exit_threshold = params["exit_threshold_nm"]
    exit_cutoff = params["exit_contact_cutoff_nm"]
    exit_window = ps_to_frames("exit_window_ps", params["exit_window_ps"], params["dt_ps"])

    out_dir = params["output_dir"]
    mindist_dir = os.path.join(out_dir, "mindist")
    exit_dir = os.path.join(out_dir, "exit_residues")
    for d in (out_dir, mindist_dir, exit_dir):
        os.makedirs(d, exist_ok=True)

    residues_per_frame = []   # one set of contacting resids per frame (all trajectories concatenated)
    frame_labels = []         # (trajectory name, local frame index) for each global frame
    boundaries = []           # global index of the first frame of each trajectory
    exit_sites = {}           # name -> exit frame, exit time and residues contacted before exit

    for name, topology, trajectory in trajectories:
        u = mda.Universe(topology, trajectory)
        target_atoms = u.select_atoms(params["target_selection"])
        ligand_atoms = u.select_atoms(params["ligand_selection"])
        exit_atoms = u.select_atoms(params["exit_selection"])
        for label, sel in (("target", target_atoms), ("ligand", ligand_atoms), ("exit", exit_atoms)):
            if len(sel) == 0:
                sys.exit(f"[{name}] the {label} selection is empty")

        n_frames = len(u.trajectory)
        print(f"[{name}] {n_frames} frames -> computing contacts (cutoff {cutoff} nm) "
              f"and ligand-protein minimum distance...")
        boundaries.append(len(residues_per_frame))
        min_dist = np.zeros(n_frames)
        time_ps = np.zeros(n_frames)

        for ts in u.trajectory:
            residues_per_frame.append(residues_within(target_atoms, ligand_atoms, cutoff, ts.dimensions))
            frame_labels.append((name, ts.frame))
            d = distance_array(exit_atoms.positions, ligand_atoms.positions, box=ts.dimensions)
            min_dist[ts.frame] = d.min() / 10.0
            time_ps[ts.frame] = ts.time

        np.savetxt(os.path.join(mindist_dir, f"mindist_{name}.dat"), np.column_stack((time_ps, min_dist)),
                   fmt="%.3f %.6f", header="time_ps min_distance_nm")

        # first exit: first frame in which the minimum distance exceeds exit_threshold_nm
        above = np.nonzero(min_dist > exit_threshold)[0]
        if len(above) == 0:
            print(f"[{name}] no exit detected (minimum distance never above {exit_threshold} nm)")
            continue
        exit_frame = int(above[0])

        # residues contacted by the ligand in the window preceding the exit
        window = []
        for frame in range(max(0, exit_frame - exit_window), exit_frame):
            u.trajectory[frame]
            window.append((frame, sorted(residues_within(exit_atoms, ligand_atoms, exit_cutoff, u.dimensions))))
        exit_residues = sorted(set().union(*(set(r) for _, r in window))) if window else []
        exit_sites[name] = {"exit_frame": exit_frame, "exit_time_ps": float(time_ps[exit_frame]),
                            "residues": exit_residues}

        with open(os.path.join(exit_dir, f"exit_residues_{name}.dat"), "w") as f:
            f.write(f"# Trajectory: {name}\n")
            f.write(f"# Exit frame: {exit_frame}, time = {time_ps[exit_frame]:.1f} ps\n")
            f.write(f"# Contact cutoff: {exit_cutoff} nm\n")
            f.write("# frame residues...\n")
            for frame, res in window:
                f.write(f"{frame} " + " ".join(map(str, res)) + "\n")
        print(f"[{name}] first exit at frame {exit_frame} ({time_ps[exit_frame]:.1f} ps)")

    with open(os.path.join(out_dir, "first_exit_times.dat"), "w") as f:
        f.write(f"# First exit: first frame with ligand-protein minimum distance > {exit_threshold} nm\n")
        f.write(f"{'#trajectory':<14}{'exit_frame':>12}{'exit_time_ps':>16}{'exit_time_ns':>14}\n")
        for name, _, _ in trajectories:
            if name in exit_sites:
                e = exit_sites[name]
                f.write(f"{name:<14}{e['exit_frame']:>12d}{e['exit_time_ps']:>16.1f}"
                        f"{e['exit_time_ps'] / 1000:>14.3f}\n")
            else:
                f.write(f"{name:<14}{'none':>12}{'none':>16}{'none':>14}\n")

    n_tot = len(residues_per_frame)
    print(f"\nTotal processed frames (all trajectories): {n_tot}")

    boundary_end = {}
    for i, b in enumerate(boundaries):
        boundary_end[b] = (boundaries[i + 1] - 1) if i + 1 < len(boundaries) else (n_tot - 1)

    cache = {
        "residues_per_frame": residues_per_frame,
        "frame_labels": frame_labels,
        "boundaries": boundaries,
        "boundary_end": boundary_end,
        "contact_cutoff_nm": cutoff,
        "trajectory_labels_order": [t[0] for t in trajectories],
        "exit_sites": exit_sites,
    }
    cache_path = os.path.join(out_dir, params["cache_file"])
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved: {cache_path}")
    print("Next step: python3 cavity_analysis.py <input file>")


if __name__ == "__main__":
    main()
