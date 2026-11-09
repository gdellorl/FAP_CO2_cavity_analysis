# CO2 Cavity and Exit-Site Analysis

Python scripts used to analyze CO2 migration and release in the molecular dynamics (MD) simulations of fatty acid photodecarboxylase described in

> Gianluca Dell'Orletta and Isabella Daidone, *A Distributed Diffusion Landscape Governs CO2 Coproduct Release from Fatty Acid Photodecarboxylase*, ChemRxiv 2026. [https://doi.org/10.26434/chemrxiv.15008533/v1](https://doi.org/10.26434/chemrxiv.15008533/v1)

The analysis identifies the transient cavities visited by CO2 inside the protein from residue-contact patterns, computes their populations and residence times, detects the first exit of CO2 from the protein, and groups the exit sites of the different trajectories into exit regions.

## Files

| File | Description |
|---|---|
| `cavity_precompute.py` | **Step 1**: reads the trajectories; computes, frame by frame, the residues in contact with CO2 and the CO2-protein minimum distance; detects the first exit of CO2 and the residues contacted just before it. |
| `cavity_analysis.py` | **Step 2**: cavity detection, clustering, populations, residence times, timeline figure and legend, and clustering of exit sites into exit regions. |
| `input.txt` | Input file used for the analysis. |

## Requirements

- Python >= 3.8
- [MDAnalysis](https://www.mdanalysis.org/) (`cavity_precompute.py` only)
- numpy, scipy, matplotlib

```bash
pip install MDAnalysis numpy scipy matplotlib
```

## Usage

```bash
python3 cavity_precompute.py input.txt
python3 cavity_analysis.py   input.txt
```

The first step reads the trajectories and is the time-consuming one; its results are stored in a cache file (`residues_cache.pkl`). The second step only reads the cache, so the detection and clustering parameters can be changed and the analysis repeated quickly. Both scripts read the same input file.

## Input file

The input file contains an optional list of parameters (`key = value`) followed by the list of trajectories:

```
# comment
key = value

[trajectories]
# name  topology  trajectory
MD1   system.pdb  run1.xtc
MD2   system.pdb  run2.xtc
```

- `name` is the label used for the trajectory in all output files and figures (it must not contain spaces).
- Relative paths are interpreted relative to the location of the input file.
- Lines starting with `#`, or text following ` # `, are comments.
- Trajectories are processed in the order in which they are listed. The hierarchical clustering may resolve ties between equal distances differently if this order is changed.

### Parameters

Parameters not given in the input file take the default values below, which are those used in the published analysis.

**Files**

| Parameter | Default | Description |
|---|---|---|
| `cache_file` | `residues_cache.pkl` | |
| `output_dir` | `.` | relative to the working directory |

**Residue contacts** (`cavity_precompute.py`)

| Parameter | Default | Description |
|---|---|---|
| `contact_cutoff_nm` | `0.4` | residue in contact with CO2 if any of its heavy atoms lies within this distance of any CO2 atom |
| `target_selection` | `(protein or resname FAD or resname C15) and not (name H* or type H)` | |
| `ligand_selection` | `resname CO2` | |

**First exit of CO2** (`cavity_precompute.py`)

| Parameter | Default | Description |
|---|---|---|
| `exit_selection` | `protein` | atoms used for the CO2-protein minimum distance |
| `exit_threshold_nm` | `0.36` | first exit = first frame in which the minimum distance exceeds this value |
| `exit_contact_cutoff_nm` | `0.32` | cutoff used to list the residues contacted by CO2 before the exit (exit site) |
| `exit_window_ps` | `100` | length of the window preceding the exit |

**Cavity detection** (`cavity_analysis.py`; time parameters must be multiples of `dt_ps`)

| Parameter | Default | Description |
|---|---|---|
| `dt_ps` | `10` | time between saved frames |
| `reference_window_ps` | `50` | window used to build the reference residue set of a candidate cavity |
| `consensus_fraction` | `0.5` | a residue belongs to the consensus set if present in at least this fraction of frames |
| `jaccard_threshold` | `0.4` | a cavity is occupied while the Jaccard index between the reference set and the contact set of the current frame is >= this value |
| `recrossing_tolerance_ps` | `20` | a cavity is left only after the Jaccard index stays below the threshold for longer than this |
| `min_visit_ps` | `100` | shorter visits are discarded |
| `min_residues_cavity` | `3` | visits with fewer consensus residues are discarded |

**Clustering of visits into cavities** (`cavity_analysis.py`)

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.36` | visits are merged if their similarity (average linkage) is >= this value |
| `jaccard_weight` | `0.5` | similarity = w · Jaccard + (1 − w) · overlap |

**Clustering of exit sites into exit regions** (`cavity_analysis.py`)

| Parameter | Default |
|---|---|
| `exit_similarity_threshold` | `0.36` |
| `exit_jaccard_weight` | `0.5` |

**Figures** (`cavity_analysis.py`)

| Parameter | Default | Description |
|---|---|---|
| `n_explicit_cavities` | `6` | cavities with an individual color; the others are shown in grey |
| `cavity_colors` | `0.12,0.47,0.71  0.68,0.78,0.91  1.00,0.50,0.05  1.00,0.73,0.47  0.17,0.63,0.17  0.60,0.87,0.54  0.84,0.15,0.16` | colors of Cavity 1, 2, ... as r,g,b in [0,1], hex codes (`#1f77b4`) or matplotlib names |
| `other_color` | `0.75,0.75,0.75` | |
| `gap_color` | `#F5E6A3` | |
| `outside_color` | `white` | |
| `outside_edge_color` | `black` | |
| `panel_order` | `name` | `name` (natural sort of the names) or `input` |
| `figsize` | `8 16.2` | width and height of the timeline figure (inches) |
| `vspace` | `0.03` | |
| `font_label` | `20` | |
| `font_tick` | `16` | |
| `legend_title` | `Cavity legend` | `none` for no title |
| `font_legend` | `18` | |
| `font_legend_title` | `20` | |
| `dpi` | `300` | |
| `timeline_file` | `cavity_timeline_panels.png` | |
| `legend_file` | `cavity_timeline_legend.png` | |

## Method

**Residue contacts.** For each frame, the set of residues (protein, FAD and C15 alkane) with at least one heavy atom within `contact_cutoff_nm` of a CO2 atom is computed.

**Cavity detection.** Starting from a frame, a reference residue set is built from the frames of a window of `reference_window_ps` (majority-vote consensus). The reference set is then followed forward in time through the Jaccard index J(A,B) = |A ∩ B| / |A ∪ B| between the reference set and the contact set of each frame. The cavity is considered occupied while J >= `jaccard_threshold`, and left only when J stays below the threshold for longer than `recrossing_tolerance_ps`. Visits shorter than `min_visit_ps`, or with fewer than `min_residues_cavity` consensus residues, are discarded. The residue set of a visit is the consensus over all its frames. Detection then restarts from the end of the visit.

**Clustering.** Visits from all trajectories are clustered by average-linkage hierarchical clustering with distance d = 1 − [w J + (1 − w) O], where O is the overlap coefficient |A ∩ B| / min(|A|, |B|). Clusters (cavities) are numbered by population, i.e. by their total residence time (Cavity 1 = most populated).

**Residence times.** The residence time of a visit is the time between its first frame and its exit frame. For each cavity, the mean, standard error of the mean (SEM), median and standard deviation over all its visits are reported.

**First exit and exit regions.** The first exit of CO2 is the first frame in which the CO2-protein minimum distance exceeds `exit_threshold_nm`. The exit site of a trajectory is the set of residues within `exit_contact_cutoff_nm` of CO2 in the `exit_window_ps` preceding the exit. Exit sites are clustered into exit regions with the same similarity measure and average linkage used for cavities.

## Output

**`cavity_precompute.py`** (in `output_dir`):

| File | Description |
|---|---|
| `residues_cache.pkl` | cache read by `cavity_analysis.py` |
| `first_exit_times.dat` | first-exit frame and time of each trajectory |
| `mindist/mindist_<name>.dat` | CO2-protein minimum distance vs time (ps, nm) |
| `exit_residues/exit_residues_<name>.dat` | residues in contact with CO2 in each frame of the window preceding the exit |

**`cavity_analysis.py`** (in `output_dir`):

| File | Description |
|---|---|
| `cavity_detection_merged.dat` | one line per visit, before clustering: trajectory, first frame, exit frame (excluded), duration (frames), consensus residues of the visit |
| `cavity_groups_merged.dat` | one line per cavity, ordered by population: index (0 = Cavity 1), number of visits, consensus residues of the cavity, and, after `\|`, its visits as `trajectory:first_frame-exit_frame` |
| `cavity_residence_summary.dat` | one line per cavity: number of visits, total residence time (ps), population (% of the total time spent in all cavities), mean, SEM, median and standard deviation of the residence times (ps) |
| `cavity_residues_readable.txt` | consensus residues of each cavity |
| `cavity_events/cavity_NNN_events.dat` | visits of cavity NNN (frames and ps) |
| `cavity_timeline_panels.png` | cavity occupancy along each trajectory |
| `cavity_timeline_legend.png` | legend of the timeline figure |
| `exit_sites_clusters.txt` | exit regions with their trajectories and residues, exit residues of each trajectory, and the pairwise similarity matrix of the exit sites |

Frame indices are 0-based and refer to the frames of each trajectory.

## How to cite

If you use these scripts, please cite:

> Gianluca Dell'Orletta and Isabella Daidone, *A Distributed Diffusion Landscape Governs CO2 Coproduct Release from Fatty Acid Photodecarboxylase*, ChemRxiv 2026. [https://doi.org/10.26434/chemrxiv.15008533/v1](https://doi.org/10.26434/chemrxiv.15008533/v1)

## License

This project is released under the MIT License (see [`LICENSE`](LICENSE)).
