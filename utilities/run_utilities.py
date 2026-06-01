import cProfile
import io
import logging
import pstats


def setup_logging(is_testing: bool) -> logging.Logger:
    """Initialise logger with verbosity based on mode."""
    logging.basicConfig(
        level=logging.DEBUG if is_testing else logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    return logging.getLogger("clustering")


def start_profiler(enabled: bool):
    if not enabled:
        return None

    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def finish_profiler(profiler):
    if profiler is None:
        return

    profiler.disable()
    stats_buffer = io.StringIO()
    pstats.Stats(profiler, stream=stats_buffer).sort_stats("cumtime").print_stats(50)
    print(stats_buffer.getvalue())
