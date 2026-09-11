#!/usr/bin/env python3
"""collect — raw lane artifacts → schema-conformant records.

Stage 2 of the three-stage contract in run.sh.template:

    run    → raw/<lane>.json        (lane-specific, verbose, keeps every sample)
    collect→ results/choreo/*.jsonl (schema/record-schema.json, one object/line)
    stats  → results/choreo/stats.json

WHY A SEPARATE STAGE. The raw artifacts carry what a reviewer needs to audit a
number — every timing sample, every stderr tail, every failed rep. The schema
records carry what the integrator needs to build a table. Collapsing the two
loses one or the other. collect.py is the lossy-but-validated projection, and it
is the ONLY place schema conformance is enforced, so a malformed lane output
fails here rather than silently poisoning stats.json.

FORMAT: JSON Lines. The schema's own $comment says "one JSON object per
observation line", so each file is newline-delimited JSON, not a JSON array.

--------------------------------------------------------------------------
ENUM COERCION — the single easiest way to get this wrong
--------------------------------------------------------------------------
Every enum in record-schema.json is a list of STRINGS, including the ones that
look boolean or numeric:

    exclusive: ["true", "false"]      level: ["1", "2"]
    flagged:   ["true", "false"]      exercised: ["true", "false"]

So Python `True` must be written as `"true"` and `1` as `"1"`. A raw `true`
JSON literal is a schema violation even though it reads as obviously correct.
`coerce()` does this and `validate()` rejects anything it missed.

--------------------------------------------------------------------------
CROSS-CUTTING PROVENANCE (manifest.md, non-negotiable)
--------------------------------------------------------------------------
"Every record carries settings_hash + kernel_hash + toolchain version." These
are NOT in the per-type `fields` lists — they are required on top of them. A
record missing any of the three is rejected, because without them the integrator
cannot tell which suite version and which binary produced the number, and the
17,353-vs-17,717 discrepancy is exactly the kind of thing that becomes
untraceable.

`kernel_hash` is null on per-CATEGORY aggregates (cost, remainder): there is no
single kernel. That is recorded explicitly rather than left to look like an
oversight.

FEEDS  every lane's records → stats.py.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/
REPO = os.path.dirname(B2)                                 # svn-artifacts/

RAW = os.path.join(HERE, "raw")
RESULTS = os.path.join(B2, "results", "choreo")
SCHEMA = os.path.join(B2, "schema", "record-schema.json")

TOOLCHAIN = "choreo"
SPEC_VERSION = "v2.1"

# The three provenance fields every record must carry (manifest.md).
PROVENANCE = ("settings_hash", "kernel_hash", "toolchain_version")

# Record types whose grain is a category, not a kernel, so kernel_hash is
# legitimately absent. Listed explicitly: a null here is a design fact, not a
# gap, and validate() must not reject it.
CATEGORY_GRAIN = {"cost", "remainder"}

# Record types that are DESIGN records, not observations. A `spec` record is one
# line of SPEC_REGISTRY: it states what the suite claims to test and whether
# that claim is expressible at all. It exists BEFORE any run, which is precisely
# its purpose -- an N/A verdict has no mutant to hang on, so the provenance rule
# ("which run produced this number?") does not apply to it.
DESIGN_GRAIN = {"spec"}

# Sources, in the order they are read. A missing source is normal (lanes run at
# different times) and is reported, not fatal — unless --require says otherwise.
# Documentation of which raw file feeds which record type; main() reads them
# explicitly. `kernel` comes from E2 ONLY — E1 deliberately contributes no kernel
# records, because mutant rows keyed by mutant_id would pollute the 310-kernel
# universe that stats.py's s4()/s5() iterate. See from_e1_mutants.
SOURCES = [
    ("mutant", "e1_mutant_records.json"),
    ("spec", "spec_registry.json"),
    ("kernel", "e2_ledger.json"),
    ("obligation", "e2_ledger.json"),
    ("cost", "e4_compile_cost.json"),
    ("residue", "e5_runtime.json"),
    ("latency", "e5_runtime.json"),
]


# --------------------------------------------------------------------------
# schema handling
# --------------------------------------------------------------------------

def load_schema():
    with open(SCHEMA) as f:
        s = json.load(f)
    return s["records"], s.get("version", "?")


def coerce(rtype, value, enums):
    """Force a value into the schema's enum vocabulary.

    Enums are string lists, so True → "true" and 1 → "1". Anything already a
    string is left alone (lowercased only for the boolean-looking ones, since
    "On" is not in ["on","off"] but is obviously meant to be).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # level is the only integral enum; keep ints as exact decimal strings.
        if float(value).is_integer():
            return str(int(value))
        return value
    if isinstance(value, str):
        if value in enums:
            return value
        low = value.lower()
        if low in enums:
            return low
    return value


def record_id(rec):
    """Best available identifier for an error message."""
    return (rec.get("mutant_id") or rec.get("kernel_id")
            or rec.get("obligation_id") or rec.get("category") or "<unknown>")


def validate(rtype, rec, spec, errors):
    """Check one record against its spec, appending human-readable errors.

    Returns nothing: the caller compares len(errors) before and after to decide
    whether this specific record was rejected.
    """
    fields = spec["fields"]
    enums = spec.get("enums", {})
    rid = record_id(rec)

    for f in fields:
        if f not in rec:
            errors.append(f"{rtype}: missing required field `{f}` (id={rid})")
    for f, allowed in enums.items():
        if f in rec and rec[f] is not None and rec[f] not in allowed:
            errors.append(f"{rtype}: field `{f}`={rec[f]!r} not in enum "
                          f"{allowed} (id={rid})")
    if rtype in DESIGN_GRAIN:
        return
    for f in PROVENANCE:
        if f not in rec:
            errors.append(f"{rtype}: missing provenance field `{f}` (id={rid})")
        elif rec[f] is None and not (f == "kernel_hash"
                                     and rtype in CATEGORY_GRAIN):
            errors.append(f"{rtype}: provenance field `{f}` is null (id={rid})")


def base_provenance(rec, ver, settings_hash=None, kernel_hash=None):
    """The three cross-cutting fields, taken from the source record where it has
    them (lanes stamp them) and filled from arguments otherwise."""
    return {
        "settings_hash": rec.get("settings_hash") or settings_hash,
        "kernel_hash": rec.get("kernel_hash") or kernel_hash,
        "toolchain_version": rec.get("toolchain_version") or ver,
    }


# --------------------------------------------------------------------------
# per-source projections
# --------------------------------------------------------------------------

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_never_causes(raw_dir):
    """mutant_id -> {cause, cause_under_execute, reason} from the never attribution.

    WHY THIS JOIN LIVES HERE. Ruling R-D2 (2026-09-10, `DECISIONS-NEEDED.md`)
    retires the old `never = 0` acceptance criterion and replaces it with
    **no *unattributed* `never`** -- every never-detected mutant must carry one of
    the five recorded causes. But `stats.py` reads ONLY the register, never
    `raw/`, so a criterion about attribution cannot be evaluated unless the
    attribution is ON the register record. Joining it here is what makes the new
    criterion checkable from committed data alone.

    The join key is `mutant_id`, verified 37/37 exact with no orphans in either
    direction. Missing file is not an error: `analyze_never.py` runs after E1, and
    collect is called between stages, so an absent attribution simply means the
    criterion is not yet evaluable (stats.py reports that explicitly rather than
    silently passing).
    """
    a = read_json(os.path.join(raw_dir, "never_attribution.json"))
    if not a:
        return {}
    out = {}
    for r in a.get("results", []):
        mid = r.get("mutant_id")
        if not mid:
            continue
        out[mid] = {"cause": r.get("cause"),
                    "cause_under_execute": r.get("cause_under_execute"),
                    "reason": r.get("reason")}
    return out


def from_e1_mutants(data, ver, never_causes=None):
    """E1 → `mutant` records (S1/S2). Returns (mutants, []) — no kernel arm.

    WHY NO `kernel` RECORDS. The first cut of this function also emitted one
    `kernel` record per mutant, keyed `kernel_id = mutant_id`, on the theory that
    "did the mutated kernel compile/run/pass" is a kernel observation. It is not,
    and emitting it is actively harmful: `kernel.jsonl` is the record space for the
    UNMUTATED suite, and stats.py's s4()/s5() iterate it to establish the
    310-kernel universe. Adding 120 mutant rows would make s4() report 120 extra
    "zero-obligation kernels" (mutants have no obligation records), destroying both
    the 310 count and the finding that the 21 zero-obligation kernels are exactly
    reshape/*. Nothing is lost by dropping the arm — the mutant record already
    carries compile_rc, run_rc, detector and oracle_passed, which is the same
    information under names that cannot be confused with the suite's.

    AUDIT FIELDS. The extras copied below are the real E1 field names
    (`detector`, `oracle_note`, `run_rc`, ...), not the ones the first draft
    guessed at (`detector_message`, `execute_rc`, `note`), which do not exist on
    an E1 record and were therefore silently copying nothing. They sit outside the
    schema's `fields` list deliberately: that list is what `validate` requires, and
    these are what make a row AUDITABLE. In particular `detector` is the field that
    separates choreo's own guard from the §7 numeric oracle — without it a reader
    cannot tell a detection from the kernel's own reference check firing, which is
    the difference between S2 = 56 and a fabricated S2 = 90.
    """
    mutants = []
    rows = data.get("records", data if isinstance(data, list) else [])
    # spec_version is a property of the MANIFEST (one generation run), not of a
    # mutant, so it is read once here and stamped onto every row.
    spec_version = data.get("spec_version") if isinstance(data, dict) else None
    for r in rows:
        # ---- v2.1 verdict, REQUIRED (specs/expansion-workflow.md §2) ------
        # A v1.0 manifest has none of these. Rather than emit a mixed corpus in
        # which some rows are path-classified and some are not -- which would
        # make S14 silently partial -- refuse it here, at the boundary, with the
        # fix in the message.
        if r.get("spec_id") is None:
            raise ValueError(
                "manifest for %r carries no `spec_id`: it predates the v2.1 "
                "path-class vocabulary. Regenerate with `gen_mutants.py` "
                "(it always stamps as_meta()). Refusing to emit a partially "
                "classified register." % (r.get("mutant_id"),))
        m = {
            "toolchain": TOOLCHAIN,
            "category": r.get("category"),
            "class": r.get("class"),
            "paper_category": r.get("paper_category"),
            "mutant_id": r.get("mutant_id"),
            "level": r.get("level", 1),
            "outcome": r.get("outcome"),
            "stage": r.get("stage"),
            "manifest": r.get("manifest"),
            "spec_id": r.get("spec_id"),
            "path_class": r.get("path_class"),
            "prohibition": r.get("prohibition") or "",
            # `applicable` is the serialization of `admissible`, renamed so the
            # record cannot be confused with the in-process `admissible` flag.
            # `admissible` is the per-MUTANT verdict; `spec_admissible` is the
            # spec's. A mutant may be inadmissible (a realized noop) even when
            # its spec is admissible, so the mutant's own verdict is the one
            # S14's denominator needs.
            "applicable": r.get("admissible", r.get("spec_admissible")),
            "spec_version": r.get("spec_version") or spec_version or SPEC_VERSION,
        }
        m.update(base_provenance(r, ver))
        # `oracle_caught` is carried alongside `detector` for the same reason:
        # it records WHICH of the three assertion sources spoke in the
        # checks-off arm. Without it a reader cannot tell whether a
        # manifest="corrupts" verdict came from the kernel's own §7 numeric
        # comparison (independent ground truth) or from choreo's ungated
        # runtime-library bounds check in ArrayProxy::operator[] (not
        # independent). Four mutants in the current E1 run are the latter -- see
        # run_e1.py's classify_log docstring for the three-source enumeration.
        for extra in ("spec", "case", "mutant_hash", "detector", "detector_passed",
                      "oracle_caught", "oracle_passed", "oracle_usable",
                      "oracle_note", "calibrated", "check_flag", "compile_rc",
                      "oracle_compile_rc", "oracle_run_rc", "run_rc", "wall_s",
                      "log_dir", "note", "infra_error"):
            if r.get(extra) is not None:
                m[extra] = r[extra]
        # R-D2: stamp the attributed cause onto every `never` row so the new
        # acceptance criterion (no unattributed never) is derivable from the
        # register. `never_cause` is None only if analyze_never.py has not run or
        # did not cover this mutant -- which stats.py then reports as a FAIL of
        # the criterion, not as a silent pass.
        if never_causes and m.get("outcome") == "never":
            c = never_causes.get(m.get("mutant_id"))
            if c:
                m["never_cause"] = c.get("cause")
                if c.get("cause_under_execute") is not None:
                    m["never_cause_under_execute"] = c["cause_under_execute"]
                if c.get("reason") is not None:
                    m["never_cause_reason"] = c["reason"]
        mutants.append(m)
    return mutants, []


def from_spec_registry(data, ver=None):
    """`raw/spec_registry.json` → `spec` records (one line per SPEC_REGISTRY entry).

    WHY THIS RECORD TYPE EXISTS. Every other record type is keyed on an
    observation. An N/A verdict has no observation -- that is what N/A means --
    so without this type the 28 exclusions are invisible in the register and the
    denominator (44 applicable / 28 N/A) is unfalsifiable. With it, a reader can
    audit the exclusion itself: `prohibition` says *why*, and for `absent` the
    `note` names the missing surface.

    It also makes `S14_path_class.per_spec` complete. A spec with no mutants
    would otherwise be silently missing from that table, which is exactly the
    failure mode the v2.1 cell floor was introduced to prevent at the mutant
    level.
    """
    out = []
    for s in data.get("specs", []):
        out.append({
            "spec_id": s.get("spec_id"),
            "class": s.get("cls") or s.get("class"),
            "path_class": s.get("path"),
            "applicable": s.get("admissible"),
            "prohibition": s.get("prohibition") or "",
            "status": s.get("status"),
            "desc": s.get("desc"),
            "note": s.get("note"),
            "spec_version": data.get("spec_version") or SPEC_VERSION,
        })
    return out


def from_e2(data, ver):
    """E2 → `kernel` records (compile arm over the unmutated suite) and one
    `obligation` record per assessed obligation (S3/S4/S5/S6)."""
    kernels, obligations = [], []
    for k in data.get("kernels", []):
        kid = k.get("kernel_id")
        cat = k.get("category")
        kernels.append({
            "toolchain": TOOLCHAIN,
            "category": cat,
            "size": k.get("size", "small"),
            "gpu_device": k.get("gpu_device"),
            "exclusive": k.get("exclusive", False),
            # E2 is compile-only: it never executes, so run/ref_check are null
            # rather than "ok". Claiming "ok" for an unexecuted kernel would be
            # a fabrication.
            "compile": k.get("compile"),
            "run": None,
            "ref_check": None,
            "kernel_id": kid,
            "shape_class": k.get("shape_class"),
            "total_obligations": k.get("total_obligations"),
            **base_provenance(k, ver),
        })
        for o in k.get("obligations", []):
            obligations.append({
                "toolchain": TOOLCHAIN,
                "category": cat,
                "kernel_id": kid,
                "obligation_id": o.get("obligation_id"),
                "class": o.get("class"),
                "outcome": o.get("outcome"),
                "mechanism": o.get("mechanism"),
                # Beyond the schema's field list, retained because S6's
                # mechanism split and the cost-filter story (plan §5.2) need
                # them and re-deriving from raw would be a second source of
                # truth.
                "source": o.get("source"),
                "cost": o.get("cost"),
                "enabled": o.get("enabled"),
                "dependence": o.get("dependence"),
                "shape_class": k.get("shape_class"),
                **base_provenance(k, ver),
            })
    return kernels, obligations


def from_e4(data, ver):
    """E4 → `cost` records (S10). Per-category grain, so kernel_hash is null."""
    out = []
    for r in data.get("records", []):
        rec = {
            "toolchain": TOOLCHAIN,
            "category": r.get("category"),
            "compile_overhead_pct": r.get("compile_overhead_pct"),
            "bucket": r.get("bucket"),
            "settings_hash": r.get("settings_hash"),
            "kernel_hash": None,          # category aggregate: no single kernel
            "toolchain_version": r.get("toolchain_version") or ver,
            # Retained: S10 is a MEDIAN, and a median without n and spread is
            # not checkable. `definition` states what was actually measured,
            # which is not the checks-on/off delta RQ4 implies (see run_e4.py).
            "n_kernels": r.get("n_kernels"),
            "min_pct": r.get("min_pct"),
            "max_pct": r.get("max_pct"),
            "mean_pct": r.get("mean_pct"),
            "definition": r.get("definition"),
            "exclusive": r.get("exclusive", False),
            "gpu_device": r.get("gpu_device"),
            "size": r.get("size"),
        }
        out.append(rec)
    return out


def from_e5(data, ver):
    """E5 → `residue` records (E5a) and `latency` records (E5b), S13."""
    residue, latency = [], []
    contended = data.get("host_contended") or []
    for r in data.get("residue_records", []):
        rec = dict(r)
        rec.setdefault("toolchain", TOOLCHAIN)
        rec["toolchain_version"] = r.get("toolchain_version") or ver
        if contended:
            rec["host_contended"] = contended
        residue.append(rec)
    for r in data.get("latency_records", []):
        rec = dict(r)
        rec.setdefault("toolchain", TOOLCHAIN)
        rec["toolchain_version"] = r.get("toolchain_version") or ver
        if contended:
            rec["host_contended"] = contended
        latency.append(rec)
    return residue, latency


# --------------------------------------------------------------------------

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="small", choices=["small", "full"],
                    help="stamped on records whose source did not carry a size")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--require", default="",
                    help="comma-separated record types that MUST be present "
                         "(default: none — a lane that has not run yet is "
                         "normal, since run.sh calls collect between stages)")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on any schema violation (default: report "
                         "and drop the offending record)")
    a = ap.parse_args()

    a.raw = os.path.abspath(a.raw)
    a.out = os.path.abspath(a.out)

    specs, schema_version = load_schema()
    print(f"[choreo] collect: raw → {os.path.relpath(a.out, REPO)} "
          f"(schema v{schema_version})")

    ver = None
    e2 = read_json(os.path.join(a.raw, "e2_ledger.json"))
    if e2:
        ver = e2.get("toolchain_version")
    if not ver:
        e4 = read_json(os.path.join(a.raw, "e4_compile_cost.json"))
        if e4:
            ver = e4.get("toolchain_version")

    buckets = {t: [] for t in specs}
    errors = []
    present, absent = [], []

    # ---- E1 → mutant (no kernel arm; see from_e1_mutants) ----
    e1 = read_json(os.path.join(a.raw, "e1_mutant_records.json"))
    if e1:
        present.append("e1_mutant_records.json")
        m, k = from_e1_mutants(e1, ver, load_never_causes(a.raw))
        buckets["mutant"].extend(m)
        buckets["kernel"].extend(k)   # always empty; kept so the tuple shape is stable
    else:
        absent.append("e1_mutant_records.json (E1 not run)")

    # ---- spec registry → spec (design records; independent of any run) ----
    sr = read_json(os.path.join(a.raw, "spec_registry.json"))
    if sr:
        present.append("spec_registry.json")
        buckets["spec"].extend(from_spec_registry(sr, ver))
    else:
        absent.append("spec_registry.json (screen not run)")

    # ---- E2 → kernel + obligation ----
    if e2:
        present.append("e2_ledger.json")
        k, o = from_e2(e2, ver)
        buckets["kernel"].extend(k)
        buckets["obligation"].extend(o)
    else:
        absent.append("e2_ledger.json (E2 not run)")

    # ---- E4 → cost ----
    e4 = read_json(os.path.join(a.raw, "e4_compile_cost.json"))
    if e4:
        present.append("e4_compile_cost.json")
        buckets["cost"].extend(from_e4(e4, ver))
    else:
        absent.append("e4_compile_cost.json (E4 not run)")

    # ---- E5 → residue + latency ----
    e5 = read_json(os.path.join(a.raw, "e5_runtime.json"))
    if e5:
        present.append("e5_runtime.json")
        r_, l_ = from_e5(e5, ver)
        buckets["residue"].extend(r_)
        buckets["latency"].extend(l_)
    else:
        absent.append("e5_runtime.json (E5 not run)")

    # ---- E3 → remainder (choreo's own remainder is folded into obligation
    #      outcomes; the `remainder` record type is the SOTA lane's, per
    #      statistics-manifest S9 and plan line 817 "e3 (SOTA)". choreo emits
    #      none, and that is correct rather than a gap.) ----

    # ------------------------------------------------------ coerce+validate ----
    clean = {}
    for rtype, recs in buckets.items():
        if rtype not in specs:
            continue
        spec = specs[rtype]
        enums = spec.get("enums", {})
        kept = []
        for rec in recs:
            for f, allowed in enums.items():
                if f in rec:
                    rec[f] = coerce(rtype, rec[f], allowed)
            before = len(errors)
            validate(rtype, rec, spec, errors)
            if len(errors) > before:
                if a.strict:
                    continue
                # Drop the malformed record but keep the error visible: a
                # silently omitted row is how a table ends up with the wrong n.
                continue
            kept.append(rec)
        clean[rtype] = kept

    # ------------------------------------------------------------- write ----
    total = 0
    print(f"\n{'record type':<16}{'written':>9}  file")
    for rtype in ("mutant", "kernel", "obligation", "cost", "residue",
                  "latency", "sanitizer", "expressibility", "remainder"):
        recs = clean.get(rtype, [])
        if not recs:
            print(f"{rtype:<16}{0:>9}  —")
            continue
        path = os.path.join(a.out, f"{rtype}.jsonl")
        n = write_jsonl(path, recs)
        total += n
        print(f"{rtype:<16}{n:>9}  {os.path.relpath(path, REPO)}")
    print(f"{'-'*16}{'-'*9}")
    print(f"{'TOTAL':<16}{total:>9}")

    if absent:
        print("\nsources not present (normal if that lane has not run yet):")
        for s in absent:
            print(f"  - {s}")

    if errors:
        # Deduplicated: one bad lane produces the same error thousands of times
        # (e.g. every obligation missing a hash), and 17k identical lines hide
        # the one distinct problem.
        uniq = {}
        for e in errors:
            key = e.split("(id=")[0]
            uniq[key] = uniq.get(key, 0) + 1
        print(f"\n*** {len(errors)} SCHEMA VIOLATION(S), "
              f"{len(uniq)} distinct — offending records DROPPED ***")
        for k, n in sorted(uniq.items(), key=lambda x: -x[1]):
            print(f"  [{n:>6}x] {k.strip()}")
    else:
        print("\nschema validation: PASS — 0 violations")

    if a.require:
        missing = [t for t in a.require.split(",") if not clean.get(t.strip())]
        if missing:
            print(f"\nERROR: --require unsatisfied, no records for: {missing}",
                  file=sys.stderr)
            return 2

    if errors and a.strict:
        return 1
    return 1 if (errors and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
