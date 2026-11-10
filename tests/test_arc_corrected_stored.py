"""Run both synthetic full-solver cases using the stored original candidate."""
import pytest
from scripts import stage_arc_corrected_intake as intake
from tests import test_catalog_arc_graph as cases


@pytest.fixture(scope='module')
def stored():
    plan=intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check(db,plan)


@pytest.mark.parametrize('count',[1,2])
def test_stored_full_solver(count,stored,tmp_path,monkeypatch):
    monkeypatch.setattr(cases,'build_catalog_arc_graph',lambda: stored)
    cases.test_full_catalog_solver(count,tmp_path,monkeypatch)
