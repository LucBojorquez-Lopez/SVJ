# HTCondor submit files (lxplus)

Translated from the SLURM scripts at the repository root.

`svj_job.sh` is tested: environment sourcing, repository resolution, `$TMPDIR`
handling and the full `scan` workflow were run locally. The `.sub` files have
**never been submitted to a real Condor pool** — smoke-test a single job before
submitting an array, per [../docs/lxplus.md](../docs/lxplus.md) §7.1.

| File | Workflow | SLURM equivalent |
|------|----------|------------------|
| `scan.sub` | parameter scan → `svj_scan_*.npz` | `run_svj_scan.sh` |
| `tsv.sub` | batch TSV shards | `run_svj_tsv.sh` |
| `validation.sub` | production validation | `run_svj_validation.sh` |
| `svj_job.sh` | shared wrapper for all three | (inline in each) |

Before first use, create the environment script `svj_job.sh` sources
(`$SVJ_ENV`, default `~/setup_env.sh`). It must source an LCG view, activate
your venv, and export `PYTHIA_DIR` / `FASTJET_DIR` — see
[../docs/lxplus.md](../docs/lxplus.md) §2.

The SLURM scripts are kept for Harvard Cannon and are unaffected.
