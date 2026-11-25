from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from commitseam.ledger import QueueLedger
from commitseam.minion_adapter import (
    ProviderSettings,
    default_fixture_path,
    diagnose_run,
    run_demo,
)

CLI_BIN = Path(sys.executable).parent / "commitseam"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    return env


def test_demo_help_works_without_credentials_and_names_fixture() -> None:
    proc = subprocess.run(
        [str(CLI_BIN), "demo", "--help"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sample_junit.xml" in proc.stdout
    assert "--yes" in proc.stdout
    assert default_fixture_path().exists()


def test_demo_non_tty_without_yes_exits_nonzero_with_guidance(tmp_path: Path) -> None:
    proc = subprocess.run(
        [str(CLI_BIN), "demo", "--provider", "offline", "--run-dir", str(tmp_path / "run")],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "--yes" in combined


def test_demo_missing_credentials_fails_cleanly_in_live_mode(tmp_path: Path) -> None:
    proc = subprocess.run(
        [str(CLI_BIN), "demo", "--yes", "--run-dir", str(tmp_path / "live-no-key")],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "GROQ_API_KEY" in combined or "OPENAI_API_KEY" in combined


def test_adapter_full_offline_flow_handoff_isolation_and_dual_crash_seams(tmp_path: Path) -> None:
    run_dir = tmp_path / "offline-demo"
    receipt = run_demo(
        run_dir=run_dir,
        fixture_path=default_fixture_path(),
        request_id="req-demo",
        provider_settings=ProviderSettings(
            provider="offline",
            model="offline-deterministic-local",
            api_key="offline-local-key",
        ),
    )

    assert receipt["mode"] == "OFFLINE_REPLAY"
    assert receipt["runtime_closed"] is True
    assert receipt["handoff_peer_read_only"] is True
    assert receipt["handoff_peer_write_tools"] == []

    # Separate SQLite files: CommitSeam queue DB vs OpenMinion session store
    queue_db = run_dir / "queue.sqlite"
    om_db = run_dir / "openminion.sqlite"
    assert queue_db.exists()
    assert om_db.exists()
    assert queue_db.resolve() != om_db.resolve()

    # Scenario 1: Approval denial -> 0 rows in rerun_jobs, denied event logged
    deny = receipt["scenarios"]["approval_denial"]
    assert deny["committed_enqueue_count"] == 0
    assert deny["denied_events_count"] >= 1
    assert deny["approval_callback_invoked"] is True

    # Scenario 2: Pre-commit crash -> 0 -> 1 across fresh session_ids
    pre = receipt["scenarios"]["pre_commit_crash"]
    assert pre["after_crash_committed_count"] == 0
    assert pre["committed_enqueue_count"] == 1
    assert pre["exact_once_ok"] is True
    assert len(set(pre["session_ids"])) >= 2

    # Scenario 3: Post-commit crash -> 1 -> 1 with identical job_id and action="reused"
    post = receipt["scenarios"]["post_commit_crash"]
    assert post["after_crash_committed_count"] == 1
    assert post["committed_enqueue_count"] == 1
    assert post["exact_once_ok"] is True
    assert post["replay_action"] == "reused"
    assert post["recovered_job_id"] == post["initial_committed_job_id"]
    assert post["created_by_attempt_id"] == post["initial_attempt_id"]
    assert len(set(post["session_ids"])) >= 2

    # Direct ledger check confirms exact-once state on disk
    ledger = QueueLedger(queue_db)
    assert len(ledger.committed_jobs("req-demo-deny")) == 0
    assert len(ledger.committed_jobs("req-demo-pre")) == 1
    assert len(ledger.committed_jobs("req-demo-post")) == 1


def test_cli_demo_and_diagnose_reconciliation_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli-run"
    demo_proc = subprocess.run(
        [
            str(CLI_BIN),
            "demo",
            "--yes",
            "--provider",
            "offline",
            "--run-dir",
            str(run_dir),
            "--request-id",
            "cli-req",
        ],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert demo_proc.returncode == 0, demo_proc.stdout + "\n" + demo_proc.stderr
    assert "MODE: OFFLINE_REPLAY" in demo_proc.stdout

    diag_proc = subprocess.run(
        [
            str(CLI_BIN),
            "diagnose",
            "--run-dir",
            str(run_dir),
            "--request-id",
            "cli-req-post",
            "--json",
        ],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert diag_proc.returncode == 0, diag_proc.stdout + "\n" + diag_proc.stderr
    report = json.loads(diag_proc.stdout)
    assert report["mode"] == "OFFLINE_REPLAY"
    assert report["request_id"] == "cli-req-post"
    assert report["committed_enqueue_count"] == 1
    assert report["exact_once_ok"] is True
    assert report["recovered_job_id"] == report["jobs"][0]["job_id"]
    assert report["memory_bundle_verified"] is True
    assert "available" in report["context_traces"]
    assert len(report["session_ids"]) >= 2

    # Programmatic diagnose_run without request_id summarizes all scenarios cleanly
    full_report = diagnose_run(run_dir)
    assert full_report["mode"] == "OFFLINE_REPLAY"
    assert set(full_report["requests"]) == {"cli-req-deny", "cli-req-pre", "cli-req-post"}


def test_review_regressions_receipt_guard_brackets_corrupt_files_and_traces(tmp_path: Path) -> None:
    import sqlite3

    # 1. --receipt-out pointing at queue.sqlite must be rejected before mutating anything
    run_dir = tmp_path / "guard-run"
    proc = subprocess.run(
        [
            str(CLI_BIN),
            "demo",
            "--yes",
            "--provider",
            "offline",
            "--run-dir",
            str(run_dir),
            "--receipt-out",
            str(run_dir / "queue.sqlite"),
        ],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 2
    assert not (run_dir / "queue.sqlite").exists()

    # 2. Parameterized pytest names with brackets survive loopback transport & pass both crash seams
    param_xml = tmp_path / "param_junit.xml"
    param_xml.write_text(
        """<testsuite name="param">
          <testcase classname="tests.t" name="test_a[x=1]"><failure message="a"/></testcase>
          <testcase classname="tests.t" name="test_b[val]"><failure message="b"/></testcase>
          <testcase classname="tests.t" name="test_c[val]"><error message="c"/></testcase>
        </testsuite>""",
        encoding="utf-8",
    )
    param_dir = tmp_path / "param-run"
    receipt = run_demo(
        run_dir=param_dir,
        fixture_path=param_xml,
        request_id="param-req",
        provider_settings=ProviderSettings(
            provider="offline",
            model="offline-deterministic-local",
            api_key="offline-local-key",
        ),
    )
    assert receipt["scenarios"]["pre_commit_crash"]["committed_enqueue_count"] == 1
    assert receipt["scenarios"]["post_commit_crash"]["replay_action"] == "reused"

    # 3. Reusing the same request_id on the same run_dir is rejected cleanly (no silent fake pass)
    dup_proc = subprocess.run(
        [
            str(CLI_BIN),
            "demo",
            "--yes",
            "--provider",
            "offline",
            "--run-dir",
            str(param_dir),
            "--request-id",
            "param-req",
        ],
        env=_clean_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert dup_proc.returncode == 2

    # 4. Seed a real OpenMinion 'context.manifest.created' event and corrupt receipts.json;
    # diagnose must still succeed on queue.sqlite, count the trace, and reject malformed bundles.
    with sqlite3.connect(param_dir / "openminion.sqlite") as conn:
        conn.execute(
            "INSERT INTO events (session_id, event_type, payload_json, created_at) VALUES (?, ?, '{}', datetime('now'))",
            ("param-req-post-s1", "context.manifest.created"),
        )
        conn.commit()
    (param_dir / "receipts.json").write_text("{corrupt json", encoding="utf-8")
    (param_dir / "memory-bundle-param-req-post.json").write_text('{"metadata": null}', encoding="utf-8")

    diag = diagnose_run(param_dir, request_id="param-req-post")
    assert diag["exact_once_ok"] is True
    assert diag["context_traces"]["available"] is True
    assert diag["context_traces"]["trace_count"] == 1
    assert diag["memory_bundle_verified"] is False
    assert "receipt_error" in diag
