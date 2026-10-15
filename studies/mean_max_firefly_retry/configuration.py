"""Preserve the frozen configuration and change only one solver tolerance."""
from dataclasses import replace
from studies.mean_max_firefly_retry.common import check_scope, NEW_RTOL
from studies.mean_max_old89.configuration import resolved as base_resolved


def resolved(row):
    check_scope(row)
    c, cameras, owners, slots, settings, geometry, old = base_resolved(row)
    settings = replace(settings, relative_tolerance=NEW_RTOL)
    assert settings.to_record() == row['semantic']['solver']
    return c, cameras, owners, slots, settings, geometry, old
