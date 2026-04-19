"""Intent classifier for user input.

Runs on CPU in ~1ms. Classifies user intent to adjust LLM parameters
(num_predict, temperature) per turn for faster responses on simple queries
and uncapped generation on complex requests.
"""

import re
from typing import NamedTuple


class Intent(NamedTuple):
    category: str        # greeting, question, story, command, followup, general
    num_predict: int     # -1 = unlimited
    temp_adjust: float   # added to base temperature


# Patterns (compiled once at import time)
_GREETING_RE = re.compile(
    r"^(hey|hi|hello|yo|sup|what'?s up|howdy|good (morning|afternoon|evening|night))"
    r"(\s+\w+)?[!?.,\s]*$",
    re.IGNORECASE,
)
_STORY_RE = re.compile(
    r"\b(tell me a story|give me a story|write me a story|make (up )?a story|"
    r"read me a story|i want a story|spin (me )?a (story|yarn|tale)|"
    r"write me|write a (story|poem|song|tale|narrative)|can you write|"
    r"explain in detail|give me a long|"
    r"describe|elaborate|go on|keep going|tell me more|"
    r"full (story|explanation))\b",
    re.IGNORECASE,
)
_SHORT_QUESTION_RE = re.compile(
    r"^(what time|what'?s the time|what day|what'?s the date|what'?s your name|"
    r"how are you|you okay|what'?s up|how old|what'?s the weather|"
    r"what'?s the temperature|what year|what month)\b",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(
    r"\b(search|look up|google|remember|forget|set a timer|play|stop|"
    r"dance|start dancing|stop dancing|read aloud|read this|"
    r"say |speak |repeat after)",
    re.IGNORECASE,
)

_MATH_RE = re.compile(
    r"\d+\s*[\+\-\*\/x×÷]\s*\d+|"
    r"\b(what('?s| is)\s+\d+\s*(plus|minus|times|divided|over|mod)\s+\d+|"
    r"what('?s| is)\s+\w+\s+(plus|minus|times|divided)\s+\w+|"
    r"calculate|compute|solve|what('?s| is) the (sum|product|difference|square root))\b",
    re.IGNORECASE,
)
_FACTUAL_RE = re.compile(
    r"(who (is|was|are|were)|what (is|are|was|were) (a|an|the) |"
    r"define |explain briefly |what does .+ mean|how (does|do) .+ work|"
    r"when (did|was|is)|where (is|are|was|were)|"
    r"(fun|interesting|cool|random) fact|tell me (a )?(fun |interesting )?fact|"
    r"did you know)",
    re.IGNORECASE,
)
_OPINION_RE = re.compile(
    r"\b(what do you think|what('?s| is) your (opinion|take|thought|view|favorite)|"
    r"how do you feel|do you (like|prefer|enjoy|love|hate)|"
    r"if you could)\b",
    re.IGNORECASE,
)


def classify_intent(user_input: str) -> Intent:
    """Classify user input intent for LLM parameter tuning.

    Returns an Intent with recommended num_predict and temp_adjust.
    """
    text = user_input.strip()

    if not text or len(text) < 3:
        return Intent("greeting", 80, 0.0)

    if _GREETING_RE.match(text):
        return Intent("greeting", 80, 0.0)

    if _SHORT_QUESTION_RE.match(text):
        return Intent("question_short", 120, 0.0)

    if _STORY_RE.search(text):
        return Intent("story", -1, 0.1)

    if _COMMAND_RE.search(text):
        return Intent("command", 200, -0.1)

    if _MATH_RE.search(text):
        return Intent("math", 80, -0.1)

    if _FACTUAL_RE.search(text):
        return Intent("factual", 150, 0.0)

    if _OPINION_RE.search(text):
        return Intent("opinion", 250, 0.05)

    # Longer input → likely needs a longer response
    word_count = len(text.split())
    if word_count > 30:
        return Intent("detailed", -1, 0.0)

    return Intent("general", 200, 0.0)
