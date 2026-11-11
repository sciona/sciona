"""Synthetic dependency graphs exercise qualification failures."""
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from scripts.review_conditional_correction_environment import dependency_closure


def audit(roots, versions, requirements, projects=None):
    def version(name):
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]
    with patch('scripts.review_conditional_correction_environment.metadata.version', side_effect=version), \
         patch('scripts.review_conditional_correction_environment.metadata.requires', side_effect=lambda name: requirements.get(name, [])):
        return dependency_closure(roots, projects or {})


def test_requested_extra_includes_missing_dependency_and_skips_inactive_marker():
    report = audit(['a[fast]'], {'a': '1'}, {'a': [
        'b; extra == "fast"', 'c; extra == "other"', 'd; python_version < "2"']})
    assert report['failures'] == [{'parent': 'a', 'requirement': 'b; extra == "fast"', 'reason': 'missing'}]
    assert not report['compatible']


def test_conflicting_edge_is_checked_even_after_package_visited():
    report = audit(['b>=2', 'a'], {'a': '1', 'b': '1'}, {'a': ['b>=1']})
    assert report['failures'] == [{'parent': None, 'requirement': 'b>=2', 'installed': '1', 'reason': 'version_conflict'}]


def test_source_manifest_overrides_stale_metadata_and_cycles_terminate():
    report = audit(['a[fast]'], {'a': '1', 'b': '2'}, {'a': ['missing'], 'b': ['a[fast]']},
                   {'a': {'dependencies': [], 'optional-dependencies': {'fast': ['b>=2']}}})
    assert report['compatible']
    assert report['versions'] == {'a': '1', 'b': '2'}


def test_unknown_source_extra_fails_closed():
    report = audit(['a[absent]'], {'a': '1'}, {}, {'a': {'dependencies': []}})
    assert report['failures'] == [{'parent': 'a', 'extra': 'absent', 'reason': 'undeclared_extra'}]
