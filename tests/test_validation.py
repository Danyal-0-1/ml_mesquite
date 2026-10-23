"""Every registered validation check passes (or skips for absent data) at quick sample sizes."""

import pytest

from mesqsim.validation import REGISTRY, run_checks


@pytest.mark.parametrize("check_id", sorted(c.id for c in REGISTRY))
def test_registered_check(check_id):
    [(chk, res)] = run_checks(quick=True, prefix=check_id)
    if res.passed is None:
        pytest.skip(f"{chk.title}: {res.result}")
    assert res.passed, f"{chk.id} {chk.title}: {res.result}"


def test_ids_unique():
    ids = [c.id for c in REGISTRY]
    assert len(ids) == len(set(ids))
