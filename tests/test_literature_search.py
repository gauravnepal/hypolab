"""Tests for LiteratureSearch."""

from unittest.mock import Mock, patch

from hypolab.literature_search import LiteratureSearch


def test_initialization():
    """Verify max_results is stored."""
    ls = LiteratureSearch(max_results=3)
    assert ls.max_results == 3


def test_build_query_with_hypothesis():
    """Verify query includes test type and variables."""
    ls = LiteratureSearch()
    hyp = {
        "test_type": "pearson_correlation",
        "variables": ["age", "income"],
    }
    query = ls._build_query(hyp)
    assert "pearson correlation" in query.lower()
    assert "age" in query
    assert "income" in query


def test_build_query_fallback():
    """Verify fallback query when hypothesis is empty."""
    ls = LiteratureSearch()
    hyp = {"test_type": "", "variables": []}
    query = ls._build_query(hyp)
    assert "statistical hypothesis testing" in query


def test_clean_text():
    """Verify whitespace normalization."""
    ls = LiteratureSearch()
    dirty = "  Hello\n\n\t  World  "
    clean = ls._clean_text(dirty)
    assert clean == "Hello World"


def test_parse_atom_empty():
    """Verify empty feed returns empty list."""
    ls = LiteratureSearch()
    result = ls._parse_atom("<feed></feed>")
    assert result == []


def test_parse_atom_valid():
    """Verify valid Atom XML is parsed correctly."""
    ls = LiteratureSearch()
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <title>Test Paper</title>
            <summary>This is a test.</summary>
            <author><name>John Doe</name></author>
            <id>http://arxiv.org/abs/1234.5678</id>
            <published>2024-01-15T00:00:00Z</published>
        </entry>
    </feed>"""
    result = ls._parse_atom(xml)
    assert len(result) == 1
    assert result[0]["title"] == "Test Paper"
    assert result[0]["authors"] == ["John Doe"]


def test_format_results_empty():
    """Verify empty list formats to 'No literature found.'"""
    ls = LiteratureSearch()
    assert ls.format_results([]) == "No literature found."


def test_format_results_with_papers():
    """Verify papers are formatted into markdown."""
    ls = LiteratureSearch()
    papers = [
        {
            "title": "T",
            "authors": ["A"],
            "published": "2024-01-01",
            "url": "http://x",
            "summary": "S",
        }
    ]
    md = ls.format_results(papers)
    assert "T" in md
    assert "A" in md


@patch("hypolab.literature_search.requests.get")
def test_search_mock(mock_get):
    """Verify search parses mocked arXiv response."""
    mock_get.return_value = Mock(
        text="""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Mock Paper</title>
                <summary>Summary here.</summary>
                <author><name>Author</name></author>
                <id>http://arxiv.org/abs/0000.0000</id>
                <published>2024-01-01T00:00:00Z</published>
            </entry>
        </feed>""",
        raise_for_status=Mock(),
    )
    ls = LiteratureSearch()
    result = ls.search("test query")
    assert len(result) == 1
    assert result[0]["title"] == "Mock Paper"
