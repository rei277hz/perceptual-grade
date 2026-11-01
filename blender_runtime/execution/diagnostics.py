"""Diagnostic values that do not keep Blender RNA alive through tracebacks."""
from dataclasses import dataclass
import traceback


def text(value):
    """Discard string subclasses that can carry native objects in attributes."""
    return ''.join((str(value),))


@dataclass(frozen=True)
class Failure:
    type_name: str
    message: str
    detail: str

    @classmethod
    def capture(cls, error):
        return cls(text(type(error).__name__), text(error), ''.join(traceback.format_exception(error)))

    def exception(self, prefix=None):
        message = self.type_name + ': ' + self.message
        error = RuntimeError((prefix + ': ' if prefix else '') + message)
        error.add_note(self.detail)
        return error


def log_exception(logger, message, *args):
    """Log diagnostic text without exporting traceback frames to handlers."""
    import sys
    error = sys.exception()
    detail = '' if error is None else Failure.capture(error).detail
    rendered = text(message % args if args else message)
    logger.error('%s\n%s', rendered, detail)
