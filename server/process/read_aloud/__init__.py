"""
Read Aloud module for Annabeth Desktop Companion.

Allows Annabeth to read selected text from any application
using her GPT-SoVITS voice, with pause/resume and Q&A support.
"""

from .text_capture import capture_selected_text, split_into_sentences, get_last_capture_debug
from .manager import ReadAloudManager, get_read_aloud_manager

__all__ = [
    'capture_selected_text',
    'get_last_capture_debug',
    'split_into_sentences',
    'ReadAloudManager',
    'get_read_aloud_manager',
]
