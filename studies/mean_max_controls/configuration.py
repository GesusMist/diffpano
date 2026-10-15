"""Replace only the mixed guidance settings in the exact validated Old89 setup."""
from dataclasses import replace
from studies.mean_max_old89.configuration import resolved as mean_max_resolved
from studies.mean_max_controls.common import MODE


def resolved(row):
    mixed=dict(row,gradient_mix_alpha=.5)
    c,cameras,owners,slots,settings,geometry,old=mean_max_resolved(mixed)
    settings=replace(settings,mode=MODE,gradient_mix_alpha=None)
    assert settings.mode=='poisson_select' and settings.gradient_mix_alpha is None
    return c,cameras,owners,slots,settings,geometry,old
