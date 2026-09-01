# HTCondor submit files (lxplus)

| File | Workflow | Merge step |
|------|----------|-----------|
| `smoke.sub` | one-point scan, `espresso` — **run this first** | — |
| `scan.sub` | parameter scan → `svj_scan_*.npz` | `scan_svj.py --merge --n-jobs N` |
| `tsv.sub` | batch TSV shards | `bash merge_svj_tsv.sh N` |
| `validation.sub` | production validation | `python merge_svj_validation.py` |
| `launch.sh` | AFS-side entry point; execs `svj_job.sh` on EOS | — |
| `svj_job.sh` | shared wrapper for all four | — |
| `_common.inc` | settings shared by all four `.sub` files | — |

## Submitting

```bash
source setup_env.sh
condor_submit condor/smoke.sub
condor_q
```

`setup_env.sh` is required: it exports the variables the `$ENV()` lookups in
`_common.inc` read at submit time (which is what keeps these files free of
hardcoded site paths), and it provisions `$SVJ_AFS` — the launcher and log
directory on AFS.

Condor cannot execute from, or write logs to, `/eos`, so `launch.sh` and the
logs live on AFS while the repository and all data stay on EOS. `launch.sh` is
tracked here and `setup_env.sh` keeps the AFS copy in sync, so edit it here.
The full reasoning is in [../docs/lxplus.md](../docs/lxplus.md) §7.2.

## Start with the smoke test

`smoke.sub` is one `espresso` job running one grid point at 2000 events, via
`src/run_regression/scan_smoke.cfg`. It takes under a minute and is the only
test that proves a worker node can read the EOS-resident repository, source
`setup_env.sh` off EOS, and write results back. Read
`$SVJ_AFS/logs/smoke_*.err` before submitting any array.

`svj_job.sh` itself — environment sourcing, repository resolution, `$TMPDIR`
handling and the full `scan` workflow — is exercised locally as well, and can
be run directly without Condor:

```bash
SVJ_SCAN_CFG=src/run_regression/scan_smoke.cfg bash condor/svj_job.sh scan 1 0
```
