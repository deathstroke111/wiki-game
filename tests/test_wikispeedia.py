import random
from pathlib import Path

import pytest

from wikigame import WikiGraph
from wikigame.data import build_wikispeedia, extract_summary

ARTICLE = """   #copyright

Bede

2007 Schools Wikipedia Selection. Related subjects: British History 1500 and
before (including Roman Britain); Historians, chroniclers and history books

   Depiction of Bede from the Nuremberg Chronicle, 1493.
   Enlarge

   Bede was an English monk at the Northumbrian monastery of Saint Peter at
   Wearmouth. He is well known as an author and scholar of early English history.

   Second paragraph that should not be used.
"""


def test_extract_summary_skips_boilerplate_and_captions():
    s = extract_summary(ARTICLE)
    assert s.startswith("Bede was an English monk")
    assert "Enlarge" not in s and "Second paragraph" not in s


def test_extract_summary_truncates_at_sentence():
    long = "This is a sentence about something. " * 40
    s = extract_summary(long, max_chars=100)
    assert len(s) <= 100 and s.endswith(".")


def test_build_from_tiny_release(tmp_path):
    g_dir, t_dir = tmp_path / "g", tmp_path / "t"
    g_dir.mkdir(), t_dir.mkdir()
    (g_dir / "articles.tsv").write_text("# c\nA%C3%A9\nB_b\n")
    (g_dir / "links.tsv").write_text("# c\nA%C3%A9\tB_b\nB_b\tA%C3%A9\nB_b\tMissing\n")
    (t_dir / "A%C3%A9.txt").write_text(ARTICLE)
    g = build_wikispeedia(g_dir, t_dir)
    assert g.titles == ["Aé", "B b"] and g.adj == [[1], [0]]  # dangling link dropped, titles decoded
    assert g.summaries[1] == ""  # missing text file tolerated


DATA = Path(__file__).resolve().parent.parent / "data" / "wikispeedia.json"


@pytest.mark.skipif(not DATA.exists(), reason="run `python -I -m wikigame.data wikispeedia` first")
def test_real_graph_stats_and_pairs():
    g = WikiGraph.load(DATA)
    assert len(g) == 4604 and 119_000 < sum(map(len, g.adj)) < 120_000
    rng = random.Random(0)
    for k in (1, 2, 3, 4):
        for _ in range(25):
            assert g.shortest_dist(*g.sample_pair(k, rng)) == k
