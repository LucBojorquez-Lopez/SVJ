"""Streaming, per-point store for raw (pre-transform) observable values.

The point of saving raw events at all is to never re-run PYTHIA: a scan writes
every observable it computed, and `fit_raw.py` re-fits any subset, with any
transform pipeline, straight from disk.  At production scale that ambition runs
into three properties of the original `--save-raw` NPZ, all of which this
module exists to fix.

**Memory.**  The NPZ path accumulated every point's events in a Python list and
`np.vstack`ed at the very end, so peak RSS was twice the whole job's raw data --
several GB for a full-grid shard, against a `request_memory` of 2 GB.  Here each
point is written to disk as it completes and then dropped, so a job's raw
footprint is one point (a few MB) no matter how large the grid.

**Redundancy.**  It stored the K grid indices as int64 *per event* (`np.tile`),
though they are constant within a point: 48 of every 176 bytes were a copy of
something already known.  Here they live in a per-*point* index.

**Precision.**  Observables are parsed from a `%e`-formatted TSV, so they carry
about seven significant digits; float64 stored fourteen.  float32 keeps every
digit the generator actually produced.

Together those take a 16-observable, 20k-event, 16384-point scan from ~58 GB to
~21 GB, and a 28-observable one to ~37 GB.

Layout, two files per scan job:

    <stem>_raw.f32        little-endian float32, C-order, (n_rows, n_obs).
                          Headerless on purpose: it is appended to point by
                          point, and a .npy header would have to know the final
                          row count up front.  Event yield per point varies with
                          acceptance, so it is not knowable up front.
    <stem>_raw.idx.npz    grid_idx  (n_points, K) int16   grid indices
                          row_start (n_points,)   int64   first row in the .f32
                          row_count (n_points,)   int32   rows for that point
                          obs_names, axis_names, n_obs

The index is rewritten at every checkpoint, so a job killed mid-flight still
leaves a readable store covering the points it finished.  Rows past the last
indexed point are ignored by the reader, which makes a truncated .f32 harmless.
"""

import os
import warnings
from pathlib import Path

import numpy as np

BIN_SUFFIX = '_raw.f32'
IDX_SUFFIX = '_raw.idx.npz'
DTYPE      = np.float32


def paths_for(stem):
    """The (binary, index) pair for a scan output stem."""
    stem = Path(stem)
    return (stem.with_name(stem.name + BIN_SUFFIX),
            stem.with_name(stem.name + IDX_SUFFIX))


class RawWriter:
    """Append raw events point by point.  Memory is O(one point)."""

    def __init__(self, stem, obs_names, axis_names, append=True):
        """Open a store, continuing an existing one by default.

        `append` is not a convenience -- opening 'wb' unconditionally destroys
        data, and did.  `scan_svj.py` resumes by skipping grid points already
        finite in its shard NPZ, so a re-run of a partly-finished shard has only
        the missing points left to do.  With a truncating open it then rewrote
        the shard containing *only* those points and discarded everything
        already stored: a backfill of 36 slices cost the Delphes stream 948 grid
        points before this was caught.  Resume semantics and a truncating writer
        are simply incompatible.

        Continuing means trusting the file over the index: only index entries
        whose rows are fully present are kept, the file is truncated to the end
        of the last such entry (dropping any partial tail from a killed job),
        and appending resumes from there.
        """
        self.bin_path, self.idx_path = paths_for(stem)
        self.obs_names  = [str(n) for n in obs_names]
        self.axis_names = [str(a) for a in axis_names]
        self.n_obs      = len(self.obs_names)
        self._grid, self._start, self._count = [], [], []
        self._rows      = 0

        resumed = False
        if append and self.idx_path.exists() and self.bin_path.exists():
            d = np.load(self.idx_path, allow_pickle=True)
            prior_obs = [str(x) for x in d['obs_names']]
            if prior_obs != self.obs_names:
                raise ValueError(
                    f"{self.idx_path.name} holds {len(prior_obs)} observables "
                    f"but this run writes {self.n_obs}; refusing to mix them")
            have_rows = self.bin_path.stat().st_size // (self.n_obs * np.dtype(DTYPE).itemsize)
            gi, st, ct = (np.array(d['grid_idx']), np.array(d['row_start']),
                          np.array(d['row_count']))
            keep = (st + ct) <= have_rows
            for g, s0, c0 in zip(gi[keep], st[keep], ct[keep]):
                self._grid.append(tuple(int(v) for v in g))
                self._start.append(int(s0))
                self._count.append(int(c0))
            self._rows = int(max((s + c for s, c in zip(self._start, self._count)),
                                 default=0))
            self._fh = open(self.bin_path, 'r+b')
            self._fh.truncate(self._rows * self.n_obs * np.dtype(DTYPE).itemsize)
            self._fh.seek(0, os.SEEK_END)
            resumed = True
        if not resumed:
            self._fh = open(self.bin_path, 'wb')
        self.resumed_points = len(self._grid)

    def append(self, gidx, raw_X):
        """Write one grid point's events.  `gidx` is the K-tuple of indices."""
        a = np.ascontiguousarray(raw_X, dtype=DTYPE)
        if a.ndim != 2 or a.shape[1] != self.n_obs:
            raise ValueError(
                f"expected (n_events, {self.n_obs}), got {a.shape}")
        a.tofile(self._fh)
        self._grid.append(tuple(int(i) for i in gidx))
        self._start.append(self._rows)
        self._count.append(a.shape[0])
        self._rows += a.shape[0]

    def flush(self):
        """Persist the index (and the binary) so far.  Safe to call often.

        The fsync is not paranoia.  On EOS the index is a small file that lands
        immediately while appended .f32 bytes are still in FUSE write-back, so
        without it the index can describe up to a point more data than another
        process can actually read -- observed live, with the index roughly three
        minutes ahead of the data file.  That is harmless for a job that
        finishes, because close() settles both; it is not harmless for a job
        killed in that window, which would leave an index promising rows that
        never arrived.  Write the data durably first, then describe it.
        """
        self._fh.flush()
        os.fsync(self._fh.fileno())
        if not self._grid:
            return
        np.savez(self.idx_path,
                 grid_idx  = np.asarray(self._grid,  dtype=np.int16),
                 row_start = np.asarray(self._start, dtype=np.int64),
                 row_count = np.asarray(self._count, dtype=np.int32),
                 obs_names = np.asarray(self.obs_names),
                 axis_names= np.asarray(self.axis_names),
                 n_obs     = np.int64(self.n_obs))

    def close(self):
        self.flush()
        self._fh.close()

    @property
    def n_rows(self):
        return self._rows

    @property
    def n_points(self):
        return len(self._grid)


class RawReader:
    """Read one or more raw stores without materialising them.

    `raw_flat` is a memmap, so slicing a point touches only that point's pages.
    Iterate with `iter_points()`; only reach for `raw_grid_flat` if some caller
    genuinely needs the per-event form, since that one does allocate.
    """

    def __init__(self, stems):
        if isinstance(stems, (str, Path)):
            stems = [stems]
        self._parts = []
        for stem in stems:
            bin_path, idx_path = paths_for(_strip_suffixes(stem))
            if not idx_path.exists():
                raise FileNotFoundError(f"no raw index at {idx_path}")
            d = np.load(idx_path, allow_pickle=True)
            n_obs = int(d['n_obs'])
            mm = np.memmap(bin_path, dtype=DTYPE, mode='r').reshape(-1, n_obs)
            grid_idx  = np.array(d['grid_idx'])
            row_start = np.array(d['row_start'])
            row_count = np.array(d['row_count'])
            # Keep only points whose rows are fully present in the file.  A
            # store being appended to on EOS can have an index momentarily
            # ahead of the data (see RawWriter.flush), and a killed job can
            # leave that state permanently.  Silently handing out rows that
            # were never written would be the worst outcome, so drop the
            # incomplete tail instead and say so.
            have = mm.shape[0]
            keep = (row_start + row_count) <= have
            if not keep.all():
                dropped = int((~keep).sum())
                warnings.warn(
                    f"{bin_path.name}: index describes {dropped} point(s) whose "
                    f"rows are not in the file; ignoring them "
                    f"(file has {have} rows). This is expected for a store "
                    f"still being written, and permanent for a killed job.",
                    stacklevel=2)
                grid_idx, row_start, row_count = (grid_idx[keep], row_start[keep],
                                                  row_count[keep])
            self._parts.append(dict(
                grid_idx=grid_idx, row_start=row_start,
                row_count=row_count, mm=mm,
                obs_names=[str(x) for x in d['obs_names']],
                axis_names=[str(x) for x in d['axis_names']]))
        if not self._parts:
            raise ValueError("no raw stores given")
        self.obs_names  = self._parts[0]['obs_names']
        self.axis_names = self._parts[0]['axis_names']
        for p in self._parts[1:]:
            if p['obs_names'] != self.obs_names:
                raise ValueError("raw shards disagree on obs_names")

    @property
    def n_points(self):
        return sum(len(p['grid_idx']) for p in self._parts)

    @property
    def n_rows(self):
        return sum(int(p['row_count'].sum()) for p in self._parts)

    def iter_points(self):
        """Yield (grid_index_tuple, events_array) for every stored point."""
        for p in self._parts:
            for gi, s, c in zip(p['grid_idx'], p['row_start'], p['row_count']):
                yield tuple(int(x) for x in gi), p['mm'][s:s + c]

    def raw_grid_flat(self):
        """Per-event grid indices, expanded.  Allocates; prefer iter_points()."""
        return np.concatenate([
            np.repeat(p['grid_idx'], p['row_count'], axis=0) for p in self._parts
        ], axis=0).astype(np.int64)

    def raw_flat(self):
        """All events, concatenated.  Allocates the whole store; prefer slicing."""
        return np.concatenate(
            [p['mm'][:int(p['row_count'].sum())] for p in self._parts], axis=0)


def _strip_suffixes(stem):
    s = str(stem)
    for suf in (IDX_SUFFIX, BIN_SUFFIX):
        if s.endswith(suf):
            return Path(s[:-len(suf)])
    return Path(s)


def exists(stem):
    return paths_for(_strip_suffixes(stem))[1].exists()
