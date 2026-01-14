"""
CommunityMed AI Utilities Module

This module contains utility functions and classes for the
CommunityMed AI TB screening system.

Components:
- impact_calculator: WHO-based impact calculations for deployment metrics
"""

from .impact_calculator import (
    ImpactConfig,
    calculate_impact,
    calculate_multi_year_impact,
    generate_report
)

__all__ = [
    "ImpactConfig",
    "calculate_impact",
    "calculate_multi_year_impact",
    "generate_report"
]
