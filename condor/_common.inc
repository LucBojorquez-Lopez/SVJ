# condor/_common.inc — settings shared by every SVJ submit file.
# Pulled in with `include : condor/_common.inc`.
#
# ── Before you submit ────────────────────────────────────────────────────────
#
#   source setup_env.sh
#   condor_submit condor/<file>.sub
#
# `setup_env.sh` exports the two variables the $ENV() lookups below read at
# submit time, and provisions $SVJ_AFS.  Without it they expand to UNDEFINED
# and the submission fails.  Keeping the paths in the environment rather than
# hardcoded here is what lets these files stay site-independent.
#
# ── Why the executable and logs are on AFS ──────────────────────────────────
#
# CERN's standard schedds reject a submit file that names any /eos path in
# exec, log, output, error or TransferInput:
#
#     Standard batch schedds cannot use /eos paths directly within the submit file.
#
# `module load lxbatch/eossubmit` routes to a schedd that accepts those, but
# that schedd then rejects every path NOT on /eos — the two cannot be mixed —
# and its transferred-back .out/.err proved unusable: they land in the EOS
# namespace but stat as `-??????????` indefinitely, so an array's stdout and
# stderr are unreadable and the jobs cannot be debugged.
#
# So the small stuff Condor itself touches lives on AFS ($SVJ_AFS: the launcher
# and the logs) and everything substantial — repository, configs, NPZs, bulk
# data — stays on EOS.  No module load, one schedd, `condor_q` just works.
#
# The worker never reads AFS: Condor transfers launch.sh to scratch.  It does
# read EOS, via the forwarded x509 proxy.

SVJ_AFS_DIR  = $ENV(SVJ_AFS)
SVJ_REPO_DIR = $ENV(SVJ_REPO)

universe   = vanilla
executable = $(SVJ_AFS_DIR)/launch.sh

# Must NOT be the EOS repo: condor_submit uses the current directory as the
# job's iwd, and an /eos iwd is one of the paths the standard schedd refuses.
# svj_job.sh cd's to $SVJ_REPO itself, so the job's own cwd is irrelevant.
initialdir = $(SVJ_AFS_DIR)

# launch.sh is a scratch copy, so it cannot infer the repository from its own
# location — SVJ_REPO is how it and svj_job.sh find everything.  Not optional.
environment = "SVJ_REPO=$(SVJ_REPO_DIR)"

# The job sources setup_env.sh itself, so it does not depend on whatever
# happened to be in the submitting shell.  A broken setup_env.sh should fail
# loudly in the job log, not be silently papered over by an inherited PATH.
getenv = False

# EOS access from the worker node.  The proxy stays on AFS.
use_x509userproxy = true
x509userproxy     = $ENV(X509_USER_PROXY)
