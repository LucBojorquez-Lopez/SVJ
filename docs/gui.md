# Interactive GUI (Jupyter)

## Quick start

Open `gui.ipynb` (or a new notebook) from the project root and run:

```python
%matplotlib widget
import sys
sys.path.insert(0, 'src/gui')
from svj_explorer import show
show()
```

The GUI loads `simulated/svj/svj_scan.npz` automatically.

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
show(n_samples=10_000)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | 10 000 | Initial number of model samples per draw |

## Observable coverage

The GUI dropdown includes all observables that are in the loaded NPZ plus any
derived observables (ratios) whose components are present. The set is determined
automatically when the module is imported.
