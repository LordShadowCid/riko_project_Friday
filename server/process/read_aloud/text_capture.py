"""
Text capture utilities for Read Aloud feature.

Captures selected text from any application using Win32 clipboard
simulation.  On Windows we use ctypes to ensure Ctrl+C is sent to
the correct foreground window (not the Annabeth overlay).
"""
import time
import re
import sys
from typing import Optional, List

import pyperclip

# ---------- Win32 helpers (Windows only) ----------
_IS_WIN = sys.platform == "win32"
if _IS_WIN:
    import ctypes
    import ctypes.wintypes
    _user32 = ctypes.windll.user32

    # Virtual-key codes
    _VK_CONTROL = 0x11
    _VK_C       = 0x43

    # SendInput structures
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_KEYUP = 0x0002

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk",         ctypes.wintypes.WORD),
            ("wScan",       ctypes.wintypes.WORD),
            ("dwFlags",     ctypes.wintypes.DWORD),
            ("time",        ctypes.wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUT(ctypes.Structure):
        class _UNION(ctypes.Union):
            _fields_ = [("ki", _KEYBDINPUT)]
        _fields_ = [
            ("type", ctypes.wintypes.DWORD),
            ("union", _UNION),
        ]

    def _send_ctrl_c() -> None:
        """Send Ctrl+C via Win32 SendInput (goes to foreground window)."""
        inputs = (_INPUT * 4)()
        # Ctrl down
        inputs[0].type = _INPUT_KEYBOARD
        inputs[0].union.ki.wVk = _VK_CONTROL
        # C down
        inputs[1].type = _INPUT_KEYBOARD
        inputs[1].union.ki.wVk = _VK_C
        # C up
        inputs[2].type = _INPUT_KEYBOARD
        inputs[2].union.ki.wVk = _VK_C
        inputs[2].union.ki.dwFlags = _KEYEVENTF_KEYUP
        # Ctrl up
        inputs[3].type = _INPUT_KEYBOARD
        inputs[3].union.ki.wVk = _VK_CONTROL
        inputs[3].union.ki.dwFlags = _KEYEVENTF_KEYUP
        _user32.SendInput(4, ctypes.pointer(inputs[0]), ctypes.sizeof(_INPUT))

    def _get_foreground_hwnd() -> int:
        return _user32.GetForegroundWindow()

    def _set_foreground(hwnd: int) -> None:
        _user32.SetForegroundWindow(hwnd)

    # Annabeth overlay hwnd – set once at startup so we can skip it
    _companion_hwnd: int = 0

    def register_companion_hwnd(hwnd: int) -> None:
        """Register the Annabeth overlay HWND so we can avoid targeting it."""
        global _companion_hwnd
        _companion_hwnd = hwnd

else:
    # Fallback for non-Windows (use pyautogui)
    import pyautogui

    def _send_ctrl_c() -> None:
        pyautogui.hotkey('ctrl', 'c')

    def _get_foreground_hwnd() -> int:
        return 0

    def _set_foreground(hwnd: int) -> None:
        pass

    _companion_hwnd = 0

    def register_companion_hwnd(hwnd: int) -> None:
        pass


def capture_selected_text(restore_clipboard: bool = True) -> Optional[str]:
    """
    Capture currently selected text from any application.

    On Windows this uses Win32 SendInput to send Ctrl+C to the
    foreground window.  If the foreground window happens to be the
    Annabeth companion overlay, the call is skipped (nothing to copy
    there) and the clipboard is checked as-is.

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

    # Determine whether we need to simulate Ctrl+C.
    # If the foreground window is the Annabeth overlay, skip the copy
    # (the overlay has no selectable text) and just read whatever is on
    # the clipboard already – the user likely copied it themselves.
    fg = _get_foreground_hwnd()
    should_copy = fg != _companion_hwnd or _companion_hwnd == 0

    if should_copy:
        # Clear clipboard to detect if copy worked
        try:
            pyperclip.copy("")
        except Exception:
            pass

        # Small delay to ensure we don't interfere with hotkey release
        time.sleep(0.05)

        # Send Ctrl+C to copy selection via Win32 / pyautogui
        _send_ctrl_c()

        # Wait for clipboard to be populated
        time.sleep(0.20)

    # Read clipboard
    try:
        text = pyperclip.paste()
    except Exception:
        text = ""

    # Restore original clipboard if requested
    if restore_clipboard and original_clipboard is not None:
        try:
            time.sleep(0.05)
            pyperclip.copy(original_clipboard)
        except Exception:
            pass

    # Return None if clipboard was empty (nothing selected)
    if not text or not text.strip():
        # Last-ditch: if we didn't copy (overlay was focused), try the
        # original clipboard content — maybe the user already copied it.
        if not should_copy and original_clipboard and original_clipboard.strip():
            return original_clipboard.strip()
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
