"""Check real PDF link annotations rather than only rendered source strings."""

from occam_rl import reporting


def test_pdf_links_exclude_sentence_punctuation(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    reporting.build_pdf("# Sources\nSee (https://example.org/card).\nhttps://example.org/A_(B)\n", [])
    pdf = (tmp_path / "output/pdf/paper.pdf").read_bytes()
    assert b"/URI (https://example.org/card)" in pdf
    assert b"/URI (https://example.org/A_\\(B\\))" in pdf
