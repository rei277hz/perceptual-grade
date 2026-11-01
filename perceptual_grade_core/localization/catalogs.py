"""Host-independent semantic messages, compiled catalogs and field labels."""
from pathlib import Path
import gettext
import json

CONTEXT = 'Perceptual Grade'
_DIRECTORY = Path(__file__).parent.parent / 'locales'
CATALOGS = {path.stem: json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(_DIRECTORY.glob('*.json'))}
if 'en' not in CATALOGS:
    raise RuntimeError('Perceptual Grade requires an English catalog')
for _locale, _messages in CATALOGS.items():
    if _messages.keys() != CATALOGS['en'].keys():
        raise RuntimeError('Perceptual Grade locale ID sets differ: ' + _locale)
# Blender renamed Simplified Chinese; both host identifiers resolve one catalog.
_TRANSLATORS = {}
for _locale in CATALOGS:
    with (_DIRECTORY / _locale / 'LC_MESSAGES' / 'perceptual_grade.mo').open('rb') as _file:
        _translator = gettext.GNUTranslations(_file)
    if any(_translator.gettext(key) != value for key, value in CATALOGS[_locale].items()):
        raise RuntimeError('Perceptual Grade compiled locale is out of date: ' + _locale)
    _TRANSLATORS[_locale] = _translator
_CORE_ERRORS = {CATALOGS['en'][message_id]: message_id for message_id in (
    'error.singular_basis', 'error.invalid_xy', 'error.nonfinite_parameter',
)}


def tr(message_id, *, locale=None, **values):
    if message_id not in CATALOGS['en']:
        raise KeyError('Unknown Perceptual Grade message ID: ' + str(message_id))
    result = _TRANSLATORS[locale or 'en'].gettext(message_id)
    return result.format(**values) if values else result


def native(message_id, **values):
    """Resolve the host's RNA source string from the English catalog."""
    return tr(message_id, locale='en', **values)


class Message(str):
    """A deferred message ID for draw code that also accepts raw numbers."""
    def __new__(cls, message_id):
        if message_id not in CATALOGS['en']:
            raise KeyError('Unknown Perceptual Grade message ID: ' + message_id)
        return super().__new__(cls, message_id)


M = Message


def error_text(value, *, locale=None):
    """Translate portable diagnostics only when presenting them in the host.

    Errors can arrive as exceptions or detached strings retained by the graph.
    Keeping those values locale independent lets a later redraw follow the
    current language, without bringing host catalogs into the numerical core.
    """
    message = ''.join((str(value),))
    prefix, separator, detail = message.rpartition(': ')
    message_id = _CORE_ERRORS.get(detail)
    return prefix + separator + tr(message_id, locale=locale) if message_id else message


def issue_text(value, *, locale=None, unknown):
    """Present only a defined issue; exception context stays in diagnostics."""
    detail = str(value).rpartition(': ')[2]
    message_id = _CORE_ERRORS.get(detail)
    return tr(message_id, locale=locale) if message_id else unknown

