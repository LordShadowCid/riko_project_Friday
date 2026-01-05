"""
Read Aloud Manager for Annabeth Desktop Companion.

Manages the read-aloud queue, pause/resume, and Q&A context.
"""
import threading
import time
from enum import Enum
from typing import Optional, List, Callable
from dataclasses import dataclass, field

from .text_capture import capture_selected_text, split_into_sentences


class ReadAloudStatus(Enum):
    """Status of the read-aloud system."""
    IDLE = "idle"           # Not reading anything
    READING = "reading"     # Currently reading
    PAUSED = "paused"       # Paused mid-reading, waiting for resume
    FINISHING = "finishing" # Finishing current sentence before pause


@dataclass
class ReadAloudState:
    """
    State container for read-aloud functionality.
    
    Thread-safe access to reading state.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    # Reading state
    _status: ReadAloudStatus = ReadAloudStatus.IDLE
    _sentences: List[str] = field(default_factory=list)
    _current_index: int = 0
    _full_text: str = ""  # Original text for Q&A context
    
    # Pause handling
    _pause_requested: bool = False
    
    @property
    def status(self) -> ReadAloudStatus:
        with self._lock:
            return self._status
    
    @status.setter
    def status(self, value: ReadAloudStatus) -> None:
        with self._lock:
            self._status = value
    
    @property
    def sentences(self) -> List[str]:
        with self._lock:
            return self._sentences.copy()
    
    @property
    def current_index(self) -> int:
        with self._lock:
            return self._current_index
    
    @current_index.setter
    def current_index(self, value: int) -> None:
        with self._lock:
            self._current_index = value
    
    @property
    def full_text(self) -> str:
        with self._lock:
            return self._full_text
    
    @property
    def pause_requested(self) -> bool:
        with self._lock:
            return self._pause_requested
    
    @pause_requested.setter
    def pause_requested(self, value: bool) -> None:
        with self._lock:
            self._pause_requested = value
    
    @property
    def is_reading(self) -> bool:
        with self._lock:
            return self._status in (ReadAloudStatus.READING, ReadAloudStatus.FINISHING)
    
    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._status == ReadAloudStatus.PAUSED
    
    @property
    def current_sentence(self) -> Optional[str]:
        """Get the current sentence being read."""
        with self._lock:
            if 0 <= self._current_index < len(self._sentences):
                return self._sentences[self._current_index]
            return None
    
    @property
    def remaining_sentences(self) -> List[str]:
        """Get sentences remaining to be read."""
        with self._lock:
            return self._sentences[self._current_index:]
    
    @property
    def sentences_read(self) -> List[str]:
        """Get sentences that have been read so far."""
        with self._lock:
            return self._sentences[:self._current_index]
    
    def start_reading(self, text: str) -> None:
        """Start reading new text."""
        sentences = split_into_sentences(text)
        with self._lock:
            self._full_text = text
            self._sentences = sentences
            self._current_index = 0
            self._pause_requested = False
            self._status = ReadAloudStatus.READING
        print(f"[ReadAloud] Starting to read {len(sentences)} sentences")
    
    def advance(self) -> Optional[str]:
        """Move to next sentence and return it, or None if done."""
        with self._lock:
            self._current_index += 1
            if self._current_index >= len(self._sentences):
                self._status = ReadAloudStatus.IDLE
                return None
            return self._sentences[self._current_index]
    
    def pause(self) -> None:
        """Request pause (will finish current sentence first)."""
        with self._lock:
            if self._status == ReadAloudStatus.READING:
                self._pause_requested = True
                self._status = ReadAloudStatus.FINISHING
        print("[ReadAloud] Pause requested - will stop after current sentence")
    
    def complete_pause(self) -> None:
        """Called when current sentence finishes, completing the pause."""
        with self._lock:
            self._pause_requested = False
            self._status = ReadAloudStatus.PAUSED
        print("[ReadAloud] Paused - you can ask questions now")
    
    def resume(self) -> Optional[str]:
        """Resume reading and return next sentence."""
        with self._lock:
            if self._status == ReadAloudStatus.PAUSED:
                self._status = ReadAloudStatus.READING
                if self._current_index < len(self._sentences):
                    return self._sentences[self._current_index]
        print("[ReadAloud] Resuming reading")
        return None
    
    def stop(self) -> None:
        """Stop reading entirely."""
        with self._lock:
            self._status = ReadAloudStatus.IDLE
            self._pause_requested = False
            self._sentences = []
            self._current_index = 0
            # Keep full_text for potential Q&A
        print("[ReadAloud] Stopped")
    
    def get_qa_context(self) -> str:
        """Get context string for Q&A about the read text."""
        with self._lock:
            if not self._full_text:
                return ""
            
            # Include what's been read so far
            read_so_far = " ".join(self._sentences[:self._current_index])
            
            return f"""I was reading the following text to you:
---
{self._full_text}
---
I've read up to: "{read_so_far}"
"""


class ReadAloudManager:
    """
    Manages the read-aloud functionality.
    
    Coordinates text capture, TTS, pause/resume, and Q&A.
    """
    
    def __init__(self):
        self.state = ReadAloudState()
        self._tts_callback: Optional[Callable[[str], float]] = None
        self._on_sentence_start: Optional[Callable[[str, int], None]] = None
        self._on_reading_complete: Optional[Callable[[], None]] = None
    
    def set_tts_callback(self, callback: Callable[[str], float]) -> None:
        """
        Set the TTS callback function.
        
        Callback should take text and return audio duration in seconds.
        """
        self._tts_callback = callback
    
    def set_sentence_callback(self, callback: Callable[[str, int], None]) -> None:
        """Set callback for when a sentence starts (for highlighting)."""
        self._on_sentence_start = callback
    
    def set_complete_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when reading completes."""
        self._on_reading_complete = callback
    
    def capture_and_read(self) -> bool:
        """
        Capture selected text and start reading it.
        
        Returns:
            True if text was captured and reading started
        """
        # Don't start if already reading
        if self.state.is_reading:
            print("[ReadAloud] Already reading - ignoring capture request")
            return False
        
        # Capture selected text
        text = capture_selected_text()
        if not text:
            print("[ReadAloud] No text selected")
            return False
        
        print(f"[ReadAloud] Captured {len(text)} characters")
        
        # Start reading
        self.state.start_reading(text)
        return True
    
    def get_next_sentence(self) -> Optional[str]:
        """
        Get the next sentence to read.
        
        Handles pause requests - returns None if pausing.
        """
        # Check if pause was requested
        if self.state.pause_requested:
            self.state.complete_pause()
            return None
        
        # Get current or advance to next
        if self.state.current_index == 0:
            sentence = self.state.current_sentence
        else:
            sentence = self.state.advance()
        
        if sentence is None:
            # Reading complete
            if self._on_reading_complete:
                self._on_reading_complete()
        else:
            # Notify of sentence start
            if self._on_sentence_start:
                self._on_sentence_start(sentence, self.state.current_index)
        
        return sentence
    
    def request_pause(self) -> None:
        """Request a pause (will finish current sentence)."""
        if self.state.is_reading:
            self.state.pause()
    
    def resume(self) -> Optional[str]:
        """Resume reading if paused."""
        if self.state.is_paused:
            return self.state.resume()
        return None
    
    def stop(self) -> None:
        """Stop reading entirely."""
        self.state.stop()
    
    def get_qa_context(self) -> str:
        """Get context for Q&A about what was read."""
        return self.state.get_qa_context()
    
    @property
    def is_active(self) -> bool:
        """Check if reading or paused (not idle)."""
        return self.state.status != ReadAloudStatus.IDLE


# Global instance
_manager: Optional[ReadAloudManager] = None


def get_read_aloud_manager() -> ReadAloudManager:
    """Get the global ReadAloudManager instance."""
    global _manager
    if _manager is None:
        _manager = ReadAloudManager()
    return _manager


if __name__ == "__main__":
    # Test the manager
    manager = get_read_aloud_manager()
    
    # Simulate reading
    test_text = "Hello there! This is a test. How are you doing today? I hope you're well."
    manager.state.start_reading(test_text)
    
    print(f"Status: {manager.state.status}")
    print(f"Sentences: {manager.state.sentences}")
    
    # Simulate reading through sentences
    while manager.state.is_reading:
        sentence = manager.get_next_sentence()
        if sentence:
            print(f"Reading: {sentence}")
            # Simulate TTS delay
            time.sleep(0.5)
            manager.state.current_index += 1
        else:
            break
    
    print(f"Final status: {manager.state.status}")
