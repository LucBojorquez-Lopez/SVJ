# Interactive GUI (Jupyter)

## Quick start

Create a notebook at the project root (notebooks are gitignored — the GUI is
driven from a scratch notebook per user) and run:

```python
%matplotlib widget
import sys
sys.path.insert(0, 'src/gui')
from svj_explorer import show
show()
```

The GUI loads the default scan automatically — `simulated/svj/working_example/`,
the same one `helpers` uses. Sliders, dropdowns and cut panels are all built
from whatever axes and observables that NPZ contains.

## GUI controls

| Control | Description |
|---------|-------------|
| **Parameter sliders** | One slider per scan axis, built dynamically from the loaded NPZ — adding or removing axes in `scan_regression.cfg` automatically updates the GUI |
| **Feature X / Y dropdowns** | Choose any two observables for the joint plot |
| **Fixed / Auto toggle** | Fixed axes use percentile-1/99 ranges from grid corners; Auto rescales to the current sample |
| **N model** | Number of model samples drawn from the interpolated distribution |
| **N validate** | Events for the VALIDATE PYTHIA run |
| **VALIDATE button** | Runs `svj_regression` at the current slider point and overlays the true distribution in crimson |
| **Cuts panel** | Per-observable range sliders to filter both model and true events; **Reset cuts** restores all |

Moving any physics slider clears the validation overlay (since the true data is now at a different point).

## `show()` arguments

```python
show(n_samples=10_000, scan_dir=None)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | 10 000 | Initial number of model samples per draw |
| `scan_dir` | None | Path to a directory containing `svj_scan.npz` and `svj_scan_meta.json`. When `None`, `helpers.DEFAULT_SCAN_DIR` (`simulated/svj/working_example/`) is used. |

To load a different scan — for instance the larger 6-axis one:

```python
show(scan_dir='simulated/svj/')          # 6 axes, 16 observables
show(scan_dir='path/to/my_scan/')        # any scan you produced yourself
```

Both `svj_scan.npz` and `svj_scan_meta.json` must be present in that directory.

## Observable coverage

The GUI dropdown includes all observables that are in the loaded NPZ plus any
derived observables (ratios) whose components are present. The set is determined
automatically when the module is imported.
