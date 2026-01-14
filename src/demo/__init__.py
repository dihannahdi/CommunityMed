"""
CommunityMed AI Demo Module

This module provides interactive demonstration interfaces
for the CommunityMed AI TB screening system.

Components:
- gradio_app: Gradio-based web interface for X-ray and cough analysis
"""

from .gradio_app import CommunityMedDemo, create_demo_interface

__all__ = ["CommunityMedDemo", "create_demo_interface"]
