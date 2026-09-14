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

STALENESS IS A SEPARATE QUESTION FROM VERSION, and it is decided by source
mtimes, not commit dates. See `newest_source_mtime` for the false alarm that
made this distinction necessary: comparing the binary against HEAD's committer
date flags a binary that was built from the checkout and committed one minute
later. `binary_stale_vs_checkout` therefore only means "some build input is
newer than the binary"; a mere version-label disagreement is reported on its
own, as `version_label_disagrees_with_head`.

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


# The checkout's build inputs. These are the ONLY things whose mtime can prove
# a binary is out of date, because they are what the build reads.
SOURCE_DIRS = ("lib", "include", "src", "tools", "specs")
SOURCE_EXT = (".cpp", ".cc", ".cxx", ".c", ".hpp", ".h", ".hxx",
              ".y", ".l", ".py")


def newest_source_mtime():
    """Newest mtime among the checkout's build inputs, or None if unreadable.

    THIS is the sound staleness evidence, and it took a false alarm to notice.
    The first version of this module compared the binary's mtime against HEAD's
    COMMITTER DATE (`mt < head_ct`). That is not the same question: an author
    who builds at 18:26 and commits at 18:27 produces a commit that postdates
    its own binary, and the test then reported a perfectly current binary as
    stale. Observed live on 2026-09-14 with ea937008 -- "choreo: reject
    non-partitioning chunkat tiles", where the checkout already held the new
    check while the run was built, and the run really did emit the new
    rejection message while simultaneously being flagged stale.

    A file mtime is causal instead of correlational: if no build input is newer
    than the binary, the binary cannot be missing a source change, no matter
    when the commit object was written.
    """
    newest = None
    for d in SOURCE_DIRS:
        root = os.path.join(CROQTILE, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Generated trees are not inputs; walking them also picks up
            # build/ inside a source dir on some layouts.
            dirnames[:] = [x for x in dirnames
                           if x not in ("build", "build-release",
                                        "build-debug", "extern",
                                        ".git", "__pycache__")]
            for fn in filenames:
                if not fn.endswith(SOURCE_EXT):
                    continue
                try:
                    m = os.path.getmtime(os.path.join(dirpath, fn))
                except OSError:
                    continue
                if newest is None or m > newest:
                    newest = m
    return newest


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

    # --- staleness ---------------------------------------------------------
    # `stale` must mean one thing only: the binary is missing a source change,
    # so the pin in manifest §5 does not describe what will run. Two candidate
    # signals, deliberately NOT merged, because they have different strengths:
    #
    #   binary_predates_source  DECISIVE. Some build input is newer than the
    #                           binary, so the binary cannot contain it.
    #   version_disagrees       ADVISORY. The mtime-inferred source commit
    #                           differs from HEAD. The inference is a heuristic
    #                           over commit dates, and a build-then-commit
    #                           sequence makes it wrong in the false-positive
    #                           direction (see newest_source_mtime). Reporting
    #                           this as `stale` is what produced a spurious
    #                           "BINARY IS STALE" on a current binary.
    newest_src = newest_source_mtime()
    binary_predates_source = (mt is not None and newest_src is not None
                              and mt < newest_src)
    version_disagrees = bool(head and src and head != src)

    if newest_src is not None:
        stale = binary_predates_source
        stale_basis = "source-mtime"
    elif mt is not None and head_ct is not None:
        # No readable sources (partial checkout, or an unbuilt tree). Fall back
        # to the commit date, and SAY it is the weaker test: this branch is the
        # one that over-reports.
        stale = mt < head_ct
        stale_basis = "commit-date-fallback"
    else:
        stale = False
        stale_basis = "unknown"

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
        "binary_stale_basis": stale_basis,
        "binary_predates_source": binary_predates_source,
        "binary_newest_source_mtime": newest_src,
        # Advisory: the version LABEL may understate HEAD. Not staleness.
        "version_label_disagrees_with_head": version_disagrees,
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
        print("[choreo]   *** BINARY IS STALE: a build input is NEWER than the "
              "binary ***", file=stream)
        print(f"[choreo]       newest source {_fmt(idt['binary_newest_source_mtime'])}"
              f" > binary {_fmt(idt['binary_mtime'])} "
              f"(test basis: {idt['binary_stale_basis']})", file=stream)
        print(f"[choreo]       records will say {idt['version']} because that "
              f"is what the binary was built from,", file=stream)
        print(f"[choreo]       NOT {idt['checkout_head']} (the checkout). "
              "Rebuild before trusting the pin", file=stream)
        print("[choreo]       in manifest §5 — but not while another worker's "
              "timing lane is mid-run:", file=stream)
        print("[choreo]       build-release/choreo is a shared path and a "
              "rebuild also breaks host-quiet.", file=stream)
    elif idt["version_label_disagrees_with_head"]:
        # Not staleness: the binary is current with every build input, but the
        # commit-date heuristic in binary_source_commit() cannot prove it. Say
        # so, because reporting this as staleness sends the reader looking for
        # a rebuild that is not needed. The label is kept rather than promoted
        # to HEAD: no source is newer than the binary, but that cannot rule out
        # a checkout that moved through commits touching only files outside
        # SOURCE_DIRS -- which is exactly the drift this module exists to
        # catch. So the label stays a conservative lower bound.
        print(f"[choreo]   note: version label {idt['version']} is older than "
              f"HEAD {idt['checkout_head']},", file=stream)
        print("[choreo]       but NO build input is newer than the binary, so "
              "this is not staleness. The label is", file=stream)
        print("[choreo]       an mtime-inferred lower bound: it understates "
              "HEAD when a build lands before its", file=stream)
        print("[choreo]       own commit. No rebuild needed; the pin is "
              "conservative, not wrong.", file=stream)
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
