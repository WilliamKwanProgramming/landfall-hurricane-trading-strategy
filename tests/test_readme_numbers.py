"""The README checker passes on the repository and fails when a quoted number is changed."""

import importlib.util
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[1]


def _checker(root: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "check_readme_numbers", REPO / "scripts" / "check_readme_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = root
    return mod


def _copy(tmp_path: pathlib.Path) -> pathlib.Path:
    shutil.copy(REPO / "README.md", tmp_path / "README.md")
    shutil.copytree(REPO / "results", tmp_path / "results")
    shutil.copytree(REPO / "docs", tmp_path / "docs")
    return tmp_path


def test_checker_passes_on_the_repository():
    assert _checker(REPO).main() == 0


def test_checker_catches_an_edited_number_a_stale_table_and_a_broken_link(tmp_path, capsys):
    root = _copy(tmp_path)
    readme = (root / "README.md").read_text()
    for old, new, message in [
        ("-0.35% [-0.63, -0.10]", "-0.30% [-0.63, -0.10]", "prose does not contain"),
        ("| HD | storm | 35 | 0.28 [-1.25, 1.77]", "| HD | storm | 35 | 0.38 [-1.25, 1.77]",
         "differs from results/tables.md"),
        ("(docs/img/placebo.png)", "(docs/img/placebo2.png)", "does not exist"),
    ]:
        assert old in readme
        (root / "README.md").write_text(readme.replace(old, new))
        assert _checker(root).main() == 1
        assert message in capsys.readouterr().out
    (root / "README.md").write_text(readme)
    loso = root / "results" / "real_eventstudy" / "trading_loso.csv"
    loso.write_text(loso.read_text().replace("-0.00351724", "-0.00300000"))
    assert _checker(root).main() == 1
