"""
render_graph.py — Utility script to export the compiled LangGraph pipeline as a PNG.

Usage:
    python render_graph.py
"""

import logging
from agentic_rag_graph import rag_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = "agentic_rag_graph.png"

executor = rag_graph.compile()
raw_graph = executor.get_graph()
raw_graph.draw_png(OUTPUT_FILE)
logger.info("Graph diagram saved to '%s'.", OUTPUT_FILE)
