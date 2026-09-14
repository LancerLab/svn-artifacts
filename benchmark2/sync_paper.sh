#!/usr/bin/env bash
# ============================================================================
# sync_paper.sh -- copy render/ output into the EuroSys paper workspace.
#
#   sync_paper.sh PAPER_DIR [--dry-run] [--force] [--full-diff]
#
# Only the floats the paper actually \input{}s are copied. render.py emits six
# additional reference-only tables (e1_*, rq1/rq3/rq5/rq6) and three generated
# PDF figures; those stay in render/ so the venue cannot accumulate dead files
# -- and, in the case of figures/fig_e1_detection.tex, cannot end up with two
# files claiming the tab:e1-detection label.
#
# Every copy is diffed first. The venue copies have been hand-edited after an
# earlier sync (notably the rq2_bugs.tex caption), so a blind `cp` would
# silently revert prose that only exists in the paper. The script therefore:
#
#   1. refuses to run if a target file holds changes that are neither
#      committed nor this script's own previous output (use --force to
#      override), because a later `git checkout` could no longer separate the
#      author's edits from the sync;
#   2. prints a status line per float (NEW / SAME / CHANGED) plus a unified
#      diff for the CHANGED ones;
#   3. copies, then prints the venue's `git diff --stat` so the exact change
#      set is reviewable and revertible.
#
# Exit codes: 0 ok, 1 a refused or failed sync, 2 usage error.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_DIR="${RENDER_DIR:-$HERE/render}"

# Floats the paper \input{}s. Keep in sync with render.py's PAPER_TABLE_FILES.
PAPER_TABLES=(
    rq2_bugs.tex
    e2_generation.tex
    e3_discharge.tex
    e4_cost.tex
    e5_oracle.tex
)
# Generated PDF figures the paper \input{}s. render.py's three figures are
# hand-drawn in the venue (fig_e1_detection/fig_plural_vn/fig_workflow are .tex
# sources), so this list is deliberately empty.
PAPER_FIGURES=()

DRY_RUN=0
FORCE=0
FULL_DIFF=0
PAPER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1 ;;
        --force)     FORCE=1 ;;
        --full-diff) FULL_DIFF=1 ;;
        -h|--help)   sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)          echo "sync_paper.sh: unknown option $1" >&2; exit 2 ;;
        *)           PAPER="$1" ;;
    esac
    shift
done

if [ -z "$PAPER" ]; then
    echo "sync_paper.sh: PAPER_DIR is required" >&2
    exit 2
fi
if [ ! -f "$PAPER/main.tex" ]; then
    echo "sync_paper.sh: '$PAPER' does not look like the paper workspace" \
         "(no main.tex)" >&2
    exit 2
fi
if [ ! -d "$RENDER_DIR/tables" ]; then
    echo "sync_paper.sh: no rendered tables at $RENDER_DIR/tables" \
         "-- run 'make render' first" >&2
    exit 2
fi

is_git() { git -C "$PAPER" rev-parse --is-inside-work-tree >/dev/null 2>&1; }

# What this script last wrote, so a second sync in the same session is not
# mistaken for an author edit. Format: `<sha256>  tables/<name>`.
STATE="$RENDER_DIR/.paper-sync-state"
sha() { sha256sum "$1" | cut -d' ' -f1; }
state_sha() {
    [ -f "$STATE" ] || return 0
    awk -v k="$1" '$2 == k { print $1; exit }' "$STATE"
}

# --- guard: never overwrite author edits -----------------------------------
# A file is safe to overwrite when it is either unchanged from HEAD or exactly
# what a previous sync wrote. Anything else is the author's own work.
if [ "$FORCE" -eq 0 ] && is_git; then
    dirty=()
    for f in "${PAPER_TABLES[@]}"; do
        rel="tables/$f"
        [ -e "$PAPER/$rel" ] || continue
        [ -n "$(git -C "$PAPER" status --porcelain -- "$rel")" ] || continue
        [ "$(sha "$PAPER/$rel")" = "$(state_sha "$rel")" ] && continue
        dirty+=("$rel")
    done
    if [ ${#dirty[@]} -gt 0 ]; then
        echo "sync_paper.sh: REFUSING -- these venue files hold changes that " >&2
        echo "               are neither committed nor a previous sync:" >&2
        printf '                 %s\n' "${dirty[@]}" >&2
        echo >&2
        echo "  Commit or stash them first (so the sync stays revertible with" >&2
        echo "  'git checkout'), or re-run with --force to overwrite." >&2
        exit 1
    fi
fi

# --- plan -----------------------------------------------------------------
declare -a STATUS=()
n_new=0; n_same=0; n_changed=0; n_missing=0
for f in "${PAPER_TABLES[@]}"; do
    src="$RENDER_DIR/tables/$f"
    dst="$PAPER/tables/$f"
    if [ ! -f "$src" ]; then
        STATUS+=("MISSING-RENDER|$f")
        n_missing=$((n_missing + 1))
    elif [ ! -f "$dst" ]; then
        STATUS+=("NEW|$f")
        n_new=$((n_new + 1))
    elif cmp -s "$src" "$dst"; then
        STATUS+=("SAME|$f")
        n_same=$((n_same + 1))
    else
        STATUS+=("CHANGED|$f")
        n_changed=$((n_changed + 1))
    fi
done

echo "[paper] $(basename "$PAPER") <- $RENDER_DIR"
for row in "${STATUS[@]}"; do
    st="${row%%|*}"; f="${row#*|}"
    printf '  %-14s tables/%s\n' "$st" "$f"
done
for f in ${PAPER_FIGURES[@]+"${PAPER_FIGURES[@]}"}; do
    src="$RENDER_DIR/figures/$f"
    dst="$PAPER/figures/$f"
    if [ -f "$src" ] && ! cmp -s "$src" "$dst"; then
        printf '  %-14s figures/%s\n' CHANGED "$f"
    fi
done
if [ "$n_missing" -gt 0 ]; then
    echo "[paper] $n_missing float(s) missing from render/ -- run 'make render'"
fi

# --- diffs ----------------------------------------------------------------
if [ "$n_changed" -gt 0 ]; then
    for row in "${STATUS[@]}"; do
        [ "${row%%|*}" = CHANGED ] || continue
        f="${row#*|}"
        echo
        echo "--- diff tables/$f (venue -> render) ---"
        # `diff` exits 1 whenever the files differ, which under `set -e` would
        # abort the sync, so its output is captured with an explicit `|| true`.
        dtext="$(diff -u "$PAPER/tables/$f" "$RENDER_DIR/tables/$f" || true)"
        nlines="$(printf '%s\n' "$dtext" | wc -l)"
        if [ "$FULL_DIFF" -eq 1 ]; then
            printf '%s\n' "$dtext"
        else
            printf '%s\n' "$dtext" | head -40
            if [ "$nlines" -gt 40 ]; then
                echo "  ... ($((nlines - 40)) more diff line(s); " \
                     "--full-diff to see all)"
            fi
        fi
    done
    echo
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[paper] dry run: nothing copied."
    exit 0
fi

# --- copy -----------------------------------------------------------------
mkdir -p "$PAPER/tables" "$PAPER/figures"
for row in "${STATUS[@]}"; do
    st="${row%%|*}"; f="${row#*|}"
    [ "$st" = MISSING-RENDER ] && continue
    cp "$RENDER_DIR/tables/$f" "$PAPER/tables/$f"
done
for f in ${PAPER_FIGURES[@]+"${PAPER_FIGURES[@]}"}; do
    cp "$RENDER_DIR/figures/$f" "$PAPER/figures/$f"
done

# render.py also writes PROSE-DRIFT.md next to its own output; leave it there.
# What the venue needs additionally is the record of what did NOT change.

# Record what we wrote, so the next sync can tell it apart from author edits.
: > "$STATE"
for row in "${STATUS[@]}"; do
    f="${row#*|}"
    dst="$PAPER/tables/$f"
    [ -f "$dst" ] && printf '%s  tables/%s\n' "$(sha "$dst")" "$f" >> "$STATE"
done
for f in ${PAPER_FIGURES[@]+"${PAPER_FIGURES[@]}"}; do
    dst="$PAPER/figures/$f"
    [ -f "$dst" ] && printf '%s  figures/%s\n' "$(sha "$dst")" "$f" >> "$STATE"
done

echo
echo "[paper] copied $n_new new / $n_changed changed / $n_same identical float(s)"
if is_git; then
    echo "[paper] review:  git -C $PAPER diff --stat -- tables figures"
    echo "[paper] revert:  git -C $PAPER checkout -- tables"
    git -C "$PAPER" --no-pager diff --stat -- tables figures || true
fi
