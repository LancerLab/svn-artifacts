#!/usr/bin/env python3
"""Toolchain identity for the choreo lane — what `toolchain_version` must mean.

--------------------------------------------------------------------------
WHY THIS MODULE EXISTS
--------------------------------------------------------------------------
manifest §5 says the choreo worker pins its version as `git rev-parse HEAD` of
the croqtile checkout. Every lane did exactly that, and on 2026-09-09 it
produced a statement that was not merely imprecise but impossible.

The timeline (all times local, all verified from mtimes and the reflog):

    09-08 20:42:03   commit 680533f lands
    09-08 23:24:18   build-release/choreo BUILT          (from 680533f)
    09-09 03:14:39   E2 + E3 ran                         checkout = 680533f
    09-09 03:21:23   E4 started                          checkout = 680533f
    09-09 03:21:49   checkout moved to f2f238f           <- 26 s into E4
    09-09 03:47:58   E4 finished
    09-09 04:11:45   E1 smoke ran                        checkout = f2f238f

E1 stamped `toolchain_version = f2f238f` on records produced by a binary built
4 h 47 m BEFORE f2f238f was committed. A binary cannot be built from a commit
that has not happened yet. The label was false, and nothing in the harness
noticed, because `git rev-parse HEAD` always succeeds and always looks
authoritative.

E2/E3/E4 escaped by luck, not by design: their `toolchain_version()` call
happened to run before 03:21:49. E4's measurement window straddled the bump —
it started 26 s earlier — so a lane that reads HEAD at start and measures for
half an hour can silently span two toolchains.

--------------------------------------------------------------------------
THE RULE
--------------------------------------------------------------------------
`toolchain_version` identifies the toolchain that PRODUCED THE DATA. What
produces data is the BINARY, not the checkout. So the version is derived from
the binary, and the checkout HEAD is recorded separately as `checkout_head` —
because a disagreement between the two is precisely the signal that a rebuild
is overdue, and hiding it inside one field would discard the only warning.

--------------------------------------------------------------------------
THE INFERENCE, AND ITS LIMITS
--------------------------------------------------------------------------
Deriving a commit from a binary is an inference, not a read, so it is kept
deliberately weak and its evidence travels with it:

    source commit = the NEWEST commit whose committer date <= the binary's mtime

Nothing stronger is available. This binary embeds no version: `strings` finds
no 40-hex git hash, no `GIT_HASH`/`BUILD_` symbol, no "choreo version" string,
and `VERSION.txt` says only "1.0.4.pre-beta" (checked 2026-09-09). mtime is
therefore the sole evidence, so `binary_mtime` and `binary_sha1_12` are
recorded on every identity dict and a reviewer can redo or reject the
inference from them.

The inference can be wrong in one direction only: if someone edits the source
and rebuilds WITHOUT committing, the binary is newer than HEAD and is reported
as HEAD. That is why `checkout_dirty` is recorded too — a dirty tree means the
binary may correspond to no commit at all.

--------------------------------------------------------------------------
WHY NOT JUST REBUILD
--------------------------------------------------------------------------
Rebuilding in place is not free and not always legal. `build-release/choreo` is
a single shared path: on 2026-09-09 another worker was 70 minutes into
`scripts/choreo_runtime_entry.py --levels none,entry --reps 5`, a timing lane
reading this very binary. Replacing it mid-run would corrupt that measurement,
and the ninja build's CPU load would violate the quiet-host condition §12.6
requires of any timing lane. So this module reports the mismatch and lets the
operator choose the moment; it never rebuilds implicitly.

Note also that the f2f238f bump was NOT cosmetic — it fixes `.pad` codegen with
`batch_dims` (lib/Target/GPU/cute_codegen.cpp), and 11 conv2d kernels use
`.pad`. None of E1's mutant bases do, so E1's RESULTS stand and only its label
was wrong. E4's `conv2d/10_dynamic` failure is a different question and is
re-examined separately, since that kernel does use `dma.pad`.
"""

import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
B2 = os.path.dirname(HERE)                 # benchmark2/
REPO = os.path.dirname(B2)                 # svn-artifacts/
CROQTILE = os.path.join(REPO, "croqtile")
BINARY = os.path.join(CROQTILE, "build-release", "choreo")


def _git(args, timeout=30):
    """Run git inside the croqtile checkout. Returns stdout, or '' on any error.

    Errors are swallowed rather than raised because identity is metadata: a lane
    must still produce records if git is unavailable, and a hard failure here
    would take down E1-E5 for a provenance nicety. Callers get '' and decide.
    """
    try:
        r = subprocess.run(["git"] + args, cwd=CROQTILE,
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def checkout_head():
    """`git rev-parse HEAD` of the croqtile checkout — what manifest §5 asks for."""
    return _git(["rev-parse", "HEAD"]) or None


def checkout_branch():
    return _git(["rev-parse", "--abbrev-ref", "HEAD"]) or None


def checkout_dirty():
    """Modified/untracked paths in the checkout, excluding build output.

    Build directories are excluded because they are ALWAYS dirty here (83 MB of
    ninja output lives inside the submodule) and would make the flag useless as
    a signal. What matters is whether *source* is dirty: that is the case where
    the binary corresponds to no commit at all.
    """
    out = _git(["status", "--porcelain"])
    if not out:
        return []
    skip = ("build-release/", "build-debug/", "build/", "extern/")
    paths = []
    for line in out.splitlines():
        p = line[3:].strip()
        if not any(p.startswith(s) for s in skip):
            paths.append(p)
    return sorted(set(paths))


def head_commit_epoch():
    """Committer date of HEAD as a unix epoch, or None."""
    s = _git(["log", "-1", "--format=%ct"])
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def binary_sha1_12():
    try:
        with open(BINARY, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def binary_mtime():
    try:
        return os.path.getmtime(BINARY)
    except OSError:
        return None


def binary_source_commit():
    """Newest commit whose committer date <= the binary's mtime.

    Returns (sha, basis) where basis explains how the answer was reached, so the
    record can carry the reasoning rather than only the conclusion. basis is one
    of:

      "mtime<=commit-date"  the normal case: binary is at least as new as this
                            commit and older than the next one
      "binary-newer-than-head"
                            binary postdates HEAD, so HEAD is the best available
                            answer — but see checkout_dirty, because an
                            uncommitted rebuild lands here too
      "unknown"             no mtime, no git, or no commit old enough
    """
    mt = binary_mtime()
    head = checkout_head()
    if mt is None or head is None:
        return (None, "unknown")

    # %ct is committer date; %H the full sha. Newest first, which is the order
    # git log gives us, so the first match walking down is the answer.
    log = _git(["log", "--format=%H%x09%ct", "-n", "200"])
    if not log:
        return (head, "binary-newer-than-head")

    for line in log.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sha, ct = parts
        try:
            ct = int(ct)
        except ValueError:
            continue
        if ct <= mt:
            if sha == head:
                return (sha, "binary-newer-than-head")
            return (sha, "mtime<=commit-date")

    # Every commit in the window postdates the binary: it was built from
    # something older than 200 commits back, or the clock is wrong.
    return (None, "unknown")


_IDENTITY_CACHE = None


def identity(refresh=False):
    """The full provenance block. Every field here is evidence, not decoration.

    `version` is the value that belongs in a record's `toolchain_version`: the
    commit the BINARY came from. `checkout_head` is recorded beside it so the
    mismatch that motivated this module stays visible instead of being resolved
    silently in favour of whichever field someone happened to read.

    MEMOIZED, deliberately. A lane asks for identity more than once — run_e5.py
    calls it at three separate points, and E5a's measurement window runs for
    minutes. Without the cache, a checkout that moved mid-run (exactly what
    happened at 03:21:49, 26 s into E4) would stamp DIFFERENT versions on
    records from the same run, and the resulting stats.json would mix two
    toolchains with no way to tell which record came from which. One run, one
    identity. Pass refresh=True to re-read, e.g. after a rebuild.
    """
    global _IDENTITY_CACHE
    if _IDENTITY_CACHE is not None and not refresh:
        return _IDENTITY_CACHE

    src, basis = binary_source_commit()
    head = checkout_head()
    mt = binary_mtime()
    head_ct = head_commit_epoch()
    dirty = checkout_dirty()

    # `stale` is the actionable flag: the checkout has moved on and the binary
    # has not been rebuilt, so the pinned version in manifest §5 no longer
    # describes what will actually run.
    stale = bool(head and src and head != src)
    if not stale and mt is not None and head_ct is not None:
        stale = mt < head_ct

    _IDENTITY_CACHE = {
        "toolchain": "choreo",
        "version": src or head,
        "version_basis": basis,
        "checkout_head": head,
        "checkout_branch": checkout_branch(),
        "checkout_dirty_source_files": dirty,
        "binary": os.path.relpath(BINARY, REPO),
        "binary_sha1_12": binary_sha1_12(),
        "binary_mtime": mt,
        "binary_stale_vs_checkout": stale,
    }
    return _IDENTITY_CACHE


def version():
    """The value for a record's `toolchain_version` field."""
    return identity()["version"]


def describe(idt=None, stream=sys.stderr):
    """Print the identity block, plus a loud warning if the binary is stale.

    Lanes call this once at startup. The warning is the whole point: the E1
    mislabel went unnoticed for a full smoke run because nothing compared the
    binary's age against the checkout's.
    """
    idt = idt or identity()
    print(f"[choreo] toolchain_version = {idt['version']}  "
          f"({idt['version_basis']})", file=stream)
    print(f"[choreo]   binary  {idt['binary']}  "
          f"sha1={idt['binary_sha1_12']}  mtime={_fmt(idt['binary_mtime'])}",
          file=stream)
    print(f"[choreo]   checkout HEAD = {idt['checkout_head']}  "
          f"branch = {idt['checkout_branch']}", file=stream)
    if idt["checkout_dirty_source_files"]:
        print(f"[choreo]   *** SOURCE TREE DIRTY: "
              f"{idt['checkout_dirty_source_files']} ***", file=stream)
        print("[choreo]       the binary may correspond to NO commit; the "
              "version above is an upper bound.", file=stream)
    if idt["binary_stale_vs_checkout"]:
        print("[choreo]   *** BINARY IS STALE: it predates the checked-out "
              "commit. ***", file=stream)
        print(f"[choreo]       records will say {idt['version']} because that "
              f"is what the binary was built from,", file=stream)
        print(f"[choreo]       NOT {idt['checkout_head']} (the checkout). "
              "Rebuild before trusting the pin", file=stream)
        print("[choreo]       in manifest §5 — but not while another worker's "
              "timing lane is mid-run:", file=stream)
        print("[choreo]       build-release/choreo is a shared path and a "
              "rebuild also breaks host-quiet.", file=stream)
    return idt


def _fmt(epoch):
    if epoch is None:
        return "?"
    import datetime
    return datetime.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


if __name__ == "__main__":
    import json
    describe()
    print(json.dumps(identity(), indent=2, default=str))
