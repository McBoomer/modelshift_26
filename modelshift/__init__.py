"""modelshift: migrate production LLM apps between models without silent regressions."""

from .capture import RecordingProvider, instrument_anthropic, instrument_openai
from .core import LLMRequest, LLMResponse
from .diff import compare
from .repair import CompositeOptimizer, HeuristicOptimizer, LLMOptimizer, repair
from .replay import replay, summarize
from .report import render_report
from .suite import build_suite, load_suite, save_suite
from .trace import Trace, TraceStore

__version__ = "0.1.0"
__all__ = [
    "LLMRequest",
    "LLMResponse",
    "Trace",
    "TraceStore",
    "instrument_openai",
    "instrument_anthropic",
    "RecordingProvider",
    "build_suite",
    "save_suite",
    "load_suite",
    "replay",
    "summarize",
    "compare",
    "repair",
    "HeuristicOptimizer",
    "LLMOptimizer",
    "CompositeOptimizer",
    "render_report",
]
