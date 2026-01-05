"""
Text capture utilities for Read Aloud feature.

Captures selected text from any application using clipboard simulation.
"""
import time
import re
from typing import Optional, List

import pyperclip
import pyautogui


def capture_selected_text(restore_clipboard: bool = True) -> Optional[str]:
    """
    Capture currently selected text from any application.
    
    Simulates Ctrl+C to copy selection to clipboard, then reads it.
    
    Args:
        restore_clipboard: If True, restore original clipboard content after
        
    Returns:
        Selected text, or None if nothing was selected
    """
    # Save original clipboard content
    original_clipboard = None
    if restore_clipboard:
        try:
            original_clipboard = pyperclip.paste()
        except Exception:
            pass  # Clipboard might be empty or contain non-text
    
    # Clear clipboard to detect if copy worked
    try:
        pyperclip.copy("")
    except Exception:
        pass
    
    # Small delay to ensure we don't interfere with hotkey release
    time.sleep(0.05)
    
    # Send Ctrl+C to copy selection
    pyautogui.hotkey('ctrl', 'c')
    
    # Wait for clipboard to be populated
    time.sleep(0.15)
    
    # Read clipboard
    try:
        text = pyperclip.paste()
    except Exception:
        text = ""
    
    # Restore original clipboard if requested
    if restore_clipboard and original_clipboard is not None:
        try:
            # Small delay before restoring
            time.sleep(0.05)
            pyperclip.copy(original_clipboard)
        except Exception:
            pass
    
    # Return None if clipboard was empty (nothing selected)
    if not text or not text.strip():
        return None
    
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for reading.
    
    Uses regex-based splitting that handles common cases well.
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple approach: split on sentence-ending punctuation followed by space and capital
    # This handles most cases well without complex lookbehind
    
    # First, protect common abbreviations by replacing their periods
    protected = text
    abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.', 'Inc.', 'Ltd.', 'Co.']
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)
    
    # Split on sentence endings followed by space and capital letter
    # Pattern: end punctuation + space + lookahead for capital
    pattern = r'([.!?]+)\s+(?=[A-Z])'
    parts = re.split(pattern, protected)
    
    # Recombine punctuation with sentences
    sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        # If next part is just punctuation, append it
        if i + 1 < len(parts) and re.match(r'^[.!?]+$', parts[i + 1]):
            sentence += parts[i + 1]
            i += 2
        else:
            i += 1
        
        if sentence:
            # Restore abbreviations
            for placeholder, abbr in placeholders.items():
                sentence = sentence.replace(placeholder, abbr)
            sentences.append(sentence)
    
    # If no splits happened, return the whole text as one sentence
    if not sentences:
        sentences = [text]
    
    return sentences


def estimate_word_timings(sentence: str, total_duration: float) -> List[tuple]:
    """
    Estimate when each word will be spoken based on total audio duration.
    
    Returns list of (word, start_time, end_time) tuples.
    
    Args:
        sentence: The sentence being spoken
        total_duration: Total audio duration in seconds
        
    Returns:
        List of (word, start_time, end_time) tuples
    """
    words = sentence.split()
    if not words:
        return []
    
    # Simple approach: equal time per word
    # Could be improved with syllable counting
    time_per_word = total_duration / len(words)
    
    timings = []
    current_time = 0.0
    
    for word in words:
        end_time = current_time + time_per_word
        timings.append((word, current_time, end_time))
        current_time = end_time
    
    return timings


if __name__ == "__main__":
    # Test sentence splitting
    test_text = """
    Dr. Smith went to the store. He bought apples, oranges, and bananas! 
    Did he forget anything? Mrs. Johnson said he might need milk.
    The price was $3.50 per item. That's expensive, isn't it?
    """
    
    sentences = split_into_sentences(test_text)
    print("Sentences found:")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
