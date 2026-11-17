"""Unified CLI runner and coverage summary reporter for `e2e_tests/`."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import unittest
from typing import Dict, List, Set, Tuple

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(E2E_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TIER_MODULES: Dict[str, Tuple[str, str]] = {
    "1": ("Tier 1: Feature Coverage", "e2e_tests.test_tier1_features"),
    "2": ("Tier 2: Boundary & Corner Cases", "e2e_tests.test_tier2_boundaries"),
    "3": ("Tier 3: Pairwise Combinations", "e2e_tests.test_tier3_pairwise"),
    "4": ("Tier 4: Workload Scenarios", "e2e_tests.test_tier4_workload"),
}

MIN_THRESHOLDS: Dict[str, int] = {
    "1": 55,
    "2": 55,
    "3": 11,
    "4": 6,
    "total": 127,
}


class CountingTextTestResult(unittest.TextTestResult):
    """TextTestResult that tracks unique test outcomes cleanly even with subTests."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.failed_ids: Set[str] = set()
        self.error_ids: Set[str] = set()
        self.skipped_ids: Set[str] = set()
        self.subtest_failures: int = 0
        self.subtest_errors: int = 0

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failed_ids.add(test.id())

    def addError(self, test, err):
        super().addError(test, err)
        self.error_ids.add(test.id())

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.skipped_ids.add(test.id())

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            if issubclass(err[0], test.failureException):
                self.failed_ids.add(test.id())
                self.subtest_failures += 1
            else:
                self.error_ids.add(test.id())
                self.subtest_errors += 1


class CountingTextTestRunner(unittest.TextTestRunner):
    resultclass = CountingTextTestResult


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Opaque-Box E2E Test Runner (Tiers 1-4) for Bilingual VSFC"
    )
    parser.add_argument(
        "--tier",
        choices=["all", "1", "2", "3", "4"],
        default="all",
        help="Run a specific tier or 'all' (default: all)",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "vi"],
        default="all",
        help="Test scope: 'all' (strict full bilingual E2E) or 'vi' (diagnostic VI-only)",
    )
    parser.add_argument(
        "--vi-only",
        action="store_true",
        help="Convenience alias for --scope vi",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose per-test output (verbosity=2)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    scope = "vi" if args.vi_only else args.scope
    os.environ["E2E_SCOPE"] = scope
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if scope == "vi":
        print(
            "[DIAGNOSTIC SCOPE=vi] English M1-M3 checks skipped; NOT valid for final M4 sign-off."
        )

    selected_tiers = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]
    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()

    summary_rows: List[Dict[str, object]] = []
    total_run = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    total_duration = 0.0

    for tier_key in selected_tiers:
        label, mod_name = TIER_MODULES[tier_key]
        print(f"\n=== Running {label} ({mod_name}) ===")
        mod = importlib.import_module(mod_name)
        suite = loader.loadTestsFromModule(mod)

        runner = CountingTextTestRunner(stream=sys.stdout, verbosity=verbosity)
        t0 = time.perf_counter()
        res: CountingTextTestResult = runner.run(suite)  # type: ignore[assignment]
        elapsed = time.perf_counter() - t0

        tests_run = res.testsRun
        failed_cnt = len(res.failed_ids)
        error_cnt = len(res.error_ids)
        skipped_cnt = len(res.skipped_ids)
        bad_ids = res.failed_ids | res.error_ids | res.skipped_ids
        passed_cnt = max(0, tests_run - len(bad_ids))

        summary_rows.append(
            {
                "tier": tier_key,
                "label": label,
                "tests_run": tests_run,
                "passed": passed_cnt,
                "failed": failed_cnt,
                "errors": error_cnt,
                "skipped": skipped_cnt,
                "duration_s": elapsed,
            }
        )
        total_run += tests_run
        total_passed += passed_cnt
        total_failed += failed_cnt
        total_errors += error_cnt
        total_skipped += skipped_cnt
        total_duration += elapsed

    print("\n" + "=" * 86)
    print(f"E2E SUMMARY (scope={scope}, tier={args.tier})")
    print("=" * 86)
    print(
        f"{'Tier':<34} {'Run':>6} {'Pass':>6} {'Fail':>6} {'Err':>6} {'Skip':>6} {'Time(s)':>9}"
    )
    print("-" * 86)
    for row in summary_rows:
        print(
            f"{row['label']:<34} {row['tests_run']:>6} {row['passed']:>6} "
            f"{row['failed']:>6} {row['errors']:>6} {row['skipped']:>6} {row['duration_s']:>9.2f}"
        )
    print("-" * 86)
    print(
        f"{'TOTAL':<34} {total_run:>6} {total_passed:>6} "
        f"{total_failed:>6} {total_errors:>6} {total_skipped:>6} {total_duration:>9.2f}"
    )
    print("=" * 86)

    threshold_ok = True
    if args.tier == "all" and scope == "all":
        for row in summary_rows:
            t_key = str(row["tier"])
            req = MIN_THRESHOLDS[t_key]
            if int(row["tests_run"]) < req:
                print(
                    f"[THRESHOLD ERROR] Tier {t_key} ran {row['tests_run']} tests, minimum required is {req}."
                )
                threshold_ok = False
        if total_run < MIN_THRESHOLDS["total"]:
            print(
                f"[THRESHOLD ERROR] Total ran {total_run} tests, minimum required is {MIN_THRESHOLDS['total']}."
            )
            threshold_ok = False
        if total_skipped > 0:
            print(
                f"[SCOPE ERROR] {total_skipped} test(s) were skipped under --scope all (0 skips required)."
            )
            threshold_ok = False

    if total_failed == 0 and total_errors == 0 and threshold_ok and (scope == "vi" or total_skipped == 0):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
