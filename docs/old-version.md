# Old version (v1)

The `old_version/` directory is a self-contained snapshot of the v1 pipeline
(12 fixed observables, Box-Cox + gennorm marginals, Gaussian copula).

```python
# From the project root:
%matplotlib widget
import sys
sys.path.insert(0, 'old_version')
from svj_explorer import show
show()
```

This loads `old_version/simulated/gennorm/gennorm_scan.npz` and uses
`old_version/helpers.py`. It cannot be re-generated from the old code
(the binary now uses the new BR parameterisation), but all GUI features work
from the saved NPZ.

See [npz-format.md](npz-format.md) for the V1 NPZ schema.
