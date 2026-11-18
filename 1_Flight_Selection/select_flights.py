"""Build the input set for the contrail phase: one converged fuel-only warm start
per flight, picked from several batch runs by directory precedence.

The fuel-only batch runs at a different initial mass per aircraft type: narrow
bodies at ``M0`` (0.80), the ``WIDEBODY_TYPES`` at ``M0_WIDEBODY`` (0.60), both from
``src/config.py``. A result solved at the other mass is refused, whatever its source. ``--mode contrails`` reads each flight's ``*_fuel.pkl.gz``
from a single ``--out-dir``, so the two populations have to be gathered into one
directory first — that is what this does.

Precedence: the FIRST ``--source`` holding a converged result for a flight wins.
List the special-case runs first and the default run last::

    python 1_Flight_Selection/select_flights.py \\
        --source 3_Results/flight_selection_2d_cruise/fuel_wide_m060 \\
        --source 3_Results/flight_selection_2d_cruise/fuel_narrow_m080 \\
        --out-dir 3_Results/flight_selection_2d_cruise/selection

Only flights that CONVERGED are copied: a non-converged warm start is the last
IPOPT iterate, not an optimum, and seeding the Pareto sweeps with it would carry
that into every point of the front.

Writes into ``--out-dir``:
    <flight_id>_sigma<σ>_fuel.pkl.gz   the selected warm starts
    flights.txt                        one flight_id per line
    manifest.csv                       flight_id, actype, m0, ipopt_hessian,
                                       fuel_kg, source — the provenance of every
                                       selected flight, since they do NOT all
                                       share one solver configuration
    rejected.csv                       with --rejected: every flight of --data-path
                                       NOT in flights.txt, and why (see
                                       rejected_flights for the reasons)

Then run the contrail sweeps over exactly that set::

    python 2_Run/batch_opentop.py --mode contrails \\
        --out-dir 3_Results/flight_selection_2d_cruise/selection \\
        --flights $(cat 3_Results/flight_selection_2d_cruise/selection/flights.txt)
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    DEFAULT_DATA_PATH, M0, M0_WIDEBODY, WIDEBODY_TYPES,
)
from src.data.flights import load_flight_index  # noqa: E402


def read_source(source: Path, sigma: float, multistart: bool = False) -> dict[str, dict]:
    """{flight_id: facts} for EVERY fuel-only result file in one directory.

    ``facts["status"]`` says what the file holds, and is the one definition of
    "converged" in this script — the selection and rejected.csv both read it, so the
    two cannot disagree about a flight:

      ``converged``      the solve ran and IPOPT returned an optimum
      ``error``          the run raised before or around the solve; the file is a stub
                         written by _run_task. Re-runnable with --retry-failed.
      ``not_converged``  the solve ran and IPOPT returned a non-optimal point.
                         Re-running it unchanged gives the same answer; it needs a
                         different m0, window or start.
      ``unreadable``     the file cannot be unpickled — typically truncated by a task
                         killed mid-write. Re-runnable with --retry-failed.
    """
    found: dict[str, dict] = {}
    suffix = "_fuel_ms" if multistart else "_fuel"   # batch_opentop.out_path_for
    for path in sorted(glob.glob(str(source / f"*_sigma{sigma:g}{suffix}.pkl.gz"))):
        name_id = Path(path).name.split("_sigma")[0]
        try:
            with gzip.open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the selection
            found[name_id] = {"path": Path(path), "actype": None, "m0": None,
                              "ipopt_hessian": None, "fuel_kg": None,
                              "alt_buffer_ft_min": None, "alt_buffer_ft_max": None,
                              "source": source.name, "status": "unreadable",
                              "detail": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        ctx = payload.get("context") or {}
        row = (payload.get("summary") or [{}])[0]
        if payload.get("error"):
            status, detail = "error", str(payload["error"])[:200]
        elif not row.get("success"):
            # The IPOPT return status, when the record carries one. Named differently
            # across modes, so try both rather than reporting a blank.
            status = "not_converged"
            detail = str(row.get("return_status") or row.get("fail_status") or "")
        else:
            status, detail = "converged", ""
        found[ctx.get("flight_id") or name_id] = {
            "path": Path(path),
            "actype": ctx.get("actype"),
            "m0": ctx.get("m0"),
            # Recorded only since the interpolant fix; older runs predate the field.
            "ipopt_hessian": ctx.get("ipopt_hessian", "unrecorded"),
            "fuel_kg": row.get("fuel_kg"),
            "alt_buffer_ft_min": ctx.get("alt_buffer_ft_min"),
            "alt_buffer_ft_max": ctx.get("alt_buffer_ft_max"),
            "source": source.name,
            "status": status,
            "detail": detail,
        }
    return found


def coverage_report(selected: dict[str, dict], dirs: list[str], sigma: float,
                    sample: int = 400) -> None:
    """Per source, how many selected flights already have a result in each
    campaign directory — and under which solver configuration those results ran.

    Answers the two questions that decide whether a late population (a retry, a
    recovered set) can be pooled with an earlier one: did the contrail phase
    actually reach these flights, and did it reach them under the SAME window,
    m0 and Hessian mode. Filenames encode none of that, so the configuration is
    read back from the pickles' own context.

    Matching is by ``<flight_id>_sigma<σ>*.pkl.gz``, so one directory covers every
    suffix a campaign uses (``_fuel``, ``_ms``, ``_dp``, ``_dp_muo``, or none).
    Read-only; the configuration counter is sampled because reading every pickle
    of a full campaign costs minutes for a number that never varies within a run.
    """
    by_source: dict[str, str] = {fid: f["source"] for fid, f in selected.items()}
    totals = Counter(by_source.values())

    for d in dirs:
        folder = Path(d)
        print(f"\n  coverage: {folder.name}")
        if not folder.is_dir():
            print("    (not a directory — skipped)")
            continue

        hits: Counter = Counter()
        suffixes: Counter = Counter()
        found: list[Path] = []
        for flight_id, source in by_source.items():
            matches = sorted(folder.glob(f"{flight_id}_sigma{sigma:g}*.pkl.gz"))
            if matches:
                hits[source] += 1
                found.extend(matches)
                for m in matches:
                    suffixes[m.name[len(flight_id):]] += 1

        for source in sorted(totals):
            n, tot = hits[source], totals[source]
            print(f"    {source:22s} {n:>5d}/{tot:<5d} ({100 * n / tot:5.1f}%)")
        if suffixes:
            print("    suffixes: " + ", ".join(f"{s}×{n}" for s, n in suffixes.most_common()))

        # Configuration actually run, read from the contexts themselves.
        cfg: Counter = Counter()
        for path in found[:sample]:
            try:
                with gzip.open(path, "rb") as fh:
                    ctx = pickle.load(fh).get("context") or {}
            except Exception:  # noqa: BLE001 — an unreadable pickle is not fatal here
                cfg["<unreadable>"] += 1
                continue
            if not ctx.get("cruise_alt_ft"):
                cfg["<error stub>"] += 1
                continue
            # Floor to whole 1000 ft: graph-2D snaps its bounds to flight levels
            # (batch_graph2d.py:238-240), so the raw difference is fractional and
            # unique per flight. Flooring recovers the buffer that was asked for —
            # and leaves a genuinely ceiling-clamped flight in its own smaller
            # bucket instead of hiding it.
            lo = ctx["cruise_alt_ft"] - ctx["min_alt_ft"]
            hi = ctx.get("max_alt_ft_requested", ctx["max_alt_ft"]) - ctx["cruise_alt_ft"]
            window = f"-{lo // 1000 * 1000:.0f}/+{hi // 1000 * 1000:.0f}"
            # The graph solvers never set ipopt_hessian; say so rather than "None".
            hessian = ctx.get("ipopt_hessian") or ("n/a" if "build_s" in ctx else "not recorded")
            cfg[f"m0={ctx.get('m0')} hessian={hessian} window={window}"] += 1
        if cfg:
            print(f"    configurations (sample of {min(len(found), sample)}):")
            for key, n in cfg.most_common():
                print(f"      {key}  ×{n}")


def rejected_flights(population: dict[str, str], selected: dict[str, dict],
                     records: list[dict[str, dict]], expected_m0) -> list[dict]:
    """One row for EVERY flight of the population that is not in flights.txt.

    The exact complement of the selection over the whole flight parquet, not just
    over the result files: a flight the sweeps never wrote a file for is as absent
    from the benchmark as one that failed, and only the parquet knows it existed.
    ``records`` is one :func:`read_source` dict per source, in precedence order.

    ``reason`` is the first that applies:

      ``wrong_mass``              converged, but at the other fleet's m0
      ``error`` / ``not_converged`` / ``unreadable``
                                    the first source's result file, see read_source
      ``no_result``                 no result file in any source: never attempted —
                                    the job hit its time limit or was killed before
                                    reaching it, or a filter (ACTYPES, FUEL_FLIGHTS)
                                    left it out
    """
    rows: list[dict] = []
    for flight_id in sorted(set(population) - set(selected)):
        actype = population[flight_id]
        want = expected_m0(actype)
        found = [rec[flight_id] for rec in records if flight_id in rec]
        converged = [f for f in found if f["status"] == "converged"]
        if converged:
            f = converged[0]
            reason = "wrong_mass"
            detail = f"solved at m0={f['m0']}, type {actype} needs m0={want:g}"
        elif found:
            f = found[0]
            reason, detail = f["status"], f["detail"]
        else:
            f = {}
            reason = "no_result"
            detail = "no result file in any source"
        rows.append({
            "flight_id": flight_id, "actype": actype, "expected_m0": f"{want:g}",
            "reason": reason, "detail": detail,
            "source": f.get("source", ""), "solved_m0": f.get("m0", ""),
            "alt_buffer_ft_min": f.get("alt_buffer_ft_min", ""),
            "alt_buffer_ft_max": f.get("alt_buffer_ft_max", ""),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True, metavar="DIR",
                   help="Batch output directory; repeat, highest precedence first.")
    p.add_argument("--out-dir", required=True,
                   help="Directory to assemble (created if missing).")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--multistart", action="store_true",
                   help="read the multistart fuel-only results (*_fuel_ms.pkl.gz, "
                        "batch_opentop --mode fuel --multistart) instead of *_fuel.pkl.gz")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the selection without writing anything.")
    p.add_argument("--coverage", action="append", metavar="DIR", default=None,
                   help="Also report, per source, how many selected flights already "
                        "have a result in DIR at this sigma (any suffix), and the m0 / "
                        "Hessian / altitude window those results ran under. Repeat for "
                        "several campaigns. Read-only — combine with --dry-run.")
    p.add_argument("--wb-types", nargs="+", default=sorted(WIDEBODY_TYPES), metavar="ICAO",
                   help="The m0 rule by aircraft type: these types may only be taken from "
                        "a result solved at --wb-m0, every other type only from one solved "
                        "at --nb-m0. Default: src.config.WIDEBODY_TYPES.")
    p.add_argument("--wb-m0", type=float, default=M0_WIDEBODY, metavar="F",
                   help=f"Mass fraction the --wb-types must have been solved at ({M0_WIDEBODY:g}).")
    p.add_argument("--nb-m0", type=float, default=M0, metavar="F",
                   help=f"Mass fraction every other type must have been solved at ({M0:g}).")
    p.add_argument("--rejected", nargs="?", const="", default=None, metavar="PATH",
                   help="Also write every flight of --data-path that is NOT in "
                        "flights.txt, with the reason, as a csv (default: "
                        "<out-dir>/rejected.csv): failed or unconverged solves, "
                        "wrong-mass results, and flights no source "
                        "has a result for at all. Combine with --dry-run to see the "
                        "counts without writing.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                   help="The flight parquet the sweeps ran on — the population "
                        "--rejected is the complement of (default: %(default)s).")
    p.add_argument("--lists-only", action="store_true",
                   help="Write flights.txt / manifest.csv / the per-m0 lists but do "
                        "NOT copy the warm-start pickles. Use this when the sweeps run "
                        "with --mode full, which re-solves the warm start in-run: the "
                        "saved one is then only a flight filter, and copying 2000+ "
                        "files achieves nothing.")
    args = p.parse_args()

    sources = [Path(s) for s in args.source]
    for src in sources:
        if not src.is_dir():
            sys.exit(f"not a directory: {src}")

    # Refuse to write into a source. The copy would target the very files being
    # read, and a stale --out-dir (an exported env var, a typo) must not be able
    # to overwrite a completed sweep.
    out_resolved = Path(args.out_dir).resolve()
    for src in sources:
        if src.resolve() == out_resolved:
            sys.exit(f"--out-dir is also a --source: {src}\n"
                     f"refusing to write results into the directory they are read from")

    # Every result file of every source, read once: the selection and rejected.csv
    # both come from these records.
    records = [read_source(src, args.sigma, args.multistart) for src in sources]

    # Walk sources in order; the first one holding a flight keeps it.
    selected: dict[str, dict] = {}
    per_source: Counter = Counter()
    shadowed: Counter = Counter()   # converged in a later source but already taken
    wb = {t.upper() for t in (args.wb_types or ())}
    wrong_mass: Counter = Counter()
    for src, rec in zip(sources, records):
        for flight_id, facts in rec.items():
            if facts["status"] != "converged":
                continue
            if wb:
                # The rule is a property of the TYPE, not of which directory happened
                # to hold the flight. A result at the other mass is not a fallback here
                # — it is a different experiment — so it is refused outright rather than
                # used when the right one is missing.
                want = args.wb_m0 if str(facts["actype"]).upper() in wb else args.nb_m0
                if facts["m0"] is None or abs(float(facts["m0"]) - want) > 1e-9:
                    wrong_mass[src.name] += 1
                    continue
            if flight_id in selected:
                shadowed[src.name] += 1
                continue
            selected[flight_id] = facts
            per_source[src.name] += 1

    if not selected:
        sys.exit("no converged flights found in any source")

    # Everything that did not make it, over the WHOLE population. Built before
    # anything is written so a --dry-run reports the same counts.
    rejected: list[dict] = []
    population: dict[str, str] = {}
    if args.rejected is not None:
        idx = load_flight_index(Path(args.data_path))
        # A missing typecode is run as a B738 by the batch (batch_opentop.run_batch);
        # the same substitution here keeps the m0 rule identical to what was solved.
        population = {fid: (str(t).upper() if isinstance(t, str) and t else "B738")
                      for fid, t in zip(idx["flight_id"], idx["typecode"])}
        rejected = rejected_flights(
            population, selected, records,
            lambda actype: args.wb_m0 if actype in wb else args.nb_m0)

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.lists_only:
            for facts in selected.values():
                shutil.copy2(facts["path"], out_dir / facts["path"].name)

        with open(out_dir / "flights.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(selected)) + "\n")

        # One list per m0, so a --mode full sweep can be submitted per mass with
        # --flights-file: the mass a flight needs is a property of the flight,
        # and --actypes can only include, never exclude.
        by_m0: dict[str, list[str]] = defaultdict(list)
        for flight_id, facts in selected.items():
            by_m0[f"{float(facts['m0']):g}"].append(flight_id)
        for m0, ids in sorted(by_m0.items()):
            name = f"flights_m0{m0.replace('.', '')}.txt"
            (out_dir / name).write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
            print(f"  wrote {name}: {len(ids)} flights at m0={m0}")

        with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["flight_id", "actype", "m0", "ipopt_hessian", "fuel_kg", "source"])
            for flight_id in sorted(selected):
                f = selected[flight_id]
                w.writerow([flight_id, f["actype"], f["m0"], f["ipopt_hessian"],
                            f["fuel_kg"], f["source"]])

    # ── report ───────────────────────────────────────────────────────────────
    print(f"{len(selected)} converged flights selected"
          f"{' (dry run — nothing copied)' if args.dry_run else f' → {out_dir}'}\n")
    if wb:
        print(f"  m0 rule: {len(wb)} widebody types at m0={args.wb_m0:g}, "
              f"everything else at m0={args.nb_m0:g}; "
              f"{sum(wrong_mass.values())} converged results refused for the wrong mass")
    for src in sources:
        extra = shadowed[src.name]
        note = f"  ({extra} also converged here but a higher-precedence source won)" if extra else ""
        print(f"  {per_source[src.name]:>5d}  {src.name}{note}")

    by_type: dict[str, Counter] = defaultdict(Counter)
    for f in selected.values():
        by_type[str(f["actype"])][str(f["m0"])] += 1
    print("\n  type    flights  m0 used")
    for actype, m0s in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        spread = ", ".join(f"{m0}×{n}" for m0, n in sorted(m0s.items()))
        print(f"  {actype:6s} {sum(m0s.values()):>8d}  {spread}")

    # The selection deliberately mixes configurations — say so loudly, because
    # the fuel-only value is the F0 the epsilon-constraint budgets derive from.
    masses = {str(f["m0"]) for f in selected.values()}
    hessians = {str(f["ipopt_hessian"]) for f in selected.values()}
    if len(masses) > 1:
        print(f"\n  NOTE: {len(masses)} different m0 values in this set ({', '.join(sorted(masses))}).")
    if len(hessians) > 1:
        print(f"  NOTE: {len(hessians)} different IPOPT Hessian modes ({', '.join(sorted(hessians))}) —"
              f"\n        fuel baselines are not directly comparable across sources.")

    if args.rejected is not None:
        print()
        print(f"  {len(population)} flights in {Path(args.data_path).name}: "
              f"{len(selected)} selected, {len(rejected)} rejected")
        by_reason: Counter = Counter(r["reason"] for r in rejected)
        for reason, n in by_reason.most_common():
            print(f"      {reason:26s} {n}")
        by_type: Counter = Counter(str(r["actype"]) for r in rejected)
        if by_type:
            print("      by type: " +
                  ", ".join(f"{t}×{n}" for t, n in by_type.most_common(12)))
        # Selected + rejected must be the whole population. A selected flight the
        # parquet does not know means the sweeps ran on a different --data-path.
        unknown = sorted(set(selected) - set(population))
        if unknown:
            print(f"  WARNING: {len(unknown)} selected flights are not in "
                  f"{args.data_path} (first: {unknown[0]}) — wrong --data-path?")
        if not args.dry_run:
            path = Path(args.rejected) if args.rejected else out_dir / "rejected.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            cols = ["flight_id", "actype", "expected_m0", "reason", "detail",
                    "source", "solved_m0", "alt_buffer_ft_min", "alt_buffer_ft_max"]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in sorted(rejected, key=lambda r: (r["reason"], r["flight_id"])):
                    w.writerow(r)
            print(f"  wrote {path}  ({len(rejected)} rows)")

    if args.coverage:
        coverage_report(selected, args.coverage, args.sigma)


if __name__ == "__main__":
    main()
