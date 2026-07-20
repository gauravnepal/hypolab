"""arXiv literature search for supporting evidence."""

import re
import warnings
from typing import Dict, List, Optional

import requests


class LiteratureSearch:
    """Search arXiv for papers related to hypothesis topics."""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    def __init__(self, max_results: int = 5):
        """Initialize with result limit."""
        self.max_results = max_results

    def search(self, query: str, start: int = 0) -> List[Dict]:
        """Query arXiv API and return paper metadata."""
        params = {
            "search_query": query,
            "start": start,
            "max_results": self.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = requests.get(self.ARXIV_API_URL, params=params, timeout=15)
            response.raise_for_status()
            return self._parse_atom(response.text)
        except requests.RequestException as e:
            warnings.warn(f"arXiv search failed: {e}")
            return []

    def search_for_hypothesis(self, hypothesis: Dict) -> List[Dict]:
        """Auto-build query from hypothesis metadata and search."""
        query = self._build_query(hypothesis)
        return self.search(query)

    def _build_query(self, hypothesis: Dict) -> str:
        """Construct arXiv search query from test type and variables."""
        test_type = hypothesis.get("test_type", "").replace("_", " ")
        variables = hypothesis.get("variables", [])
        parts = [test_type] + variables
        query = " AND ".join(f'"{p}"' for p in parts if p)
        if not query:
            query = "statistical hypothesis testing"
        return query

    def _parse_atom(self, xml_text: str) -> List[Dict]:
        """Parse arXiv Atom XML into structured dicts."""
        import xml.etree.ElementTree as ET
        
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        for entry in root.findall("atom:entry", ns):
            paper = {}
            title = entry.find("atom:title", ns)
            paper["title"] = self._clean_text(title.text) if title is not None else "N/A"
            
            summary = entry.find("atom:summary", ns)
            paper["summary"] = self._clean_text(summary.text) if summary is not None else "N/A"
            
            authors = entry.findall("atom:author/atom:name", ns)
            paper["authors"] = [a.text for a in authors if a.text]
            
            link = entry.find("atom:id", ns)
            paper["url"] = link.text if link is not None else ""
            
            published = entry.find("atom:published", ns)
            paper["published"] = published.text[:10] if published is not None else ""
            
            papers.append(paper)
        
        return papers

    def _clean_text(self, text: Optional[str]) -> str:
        """Normalize whitespace in arXiv text."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def format_results(self, papers: List[Dict]) -> str:
        """Format papers into readable markdown."""
        if not papers:
            return "No literature found."
        lines = ["### Supporting Literature\n"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"**{i}. {paper['title']}**")
            lines.append(f"   Authors: {', '.join(paper['authors'][:3])}")
            lines.append(f"   Published: {paper['published']} | [Link]({paper['url']})")
            lines.append(f"   > {paper['summary'][:200]}...\n")
        return "\n".join(lines)