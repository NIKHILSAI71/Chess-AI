"""
Knowledge Integration Module
============================

Chess knowledge components including opening books, endgame tablebases,
and position pattern recognition.
"""

from .opening_book import OpeningBook, PolyglotBook
from .tablebase import SyzygyTablebase, TablebaseProber

__all__ = [
    'OpeningBook',
    'PolyglotBook', 
    'SyzygyTablebase',
    'TablebaseProber'
]
