import json
from pathlib import Path

from django.conf import settings

from .CommonController import json_response_ensure_ascii


LOG_DIR = settings.BASE_DIR / 'chat-sd-logs'


def _get_log_path(file_name):
    if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
        return None
    log_dir = LOG_DIR.resolve()
    path = (log_dir / file_name).resolve()
    if path.parent != log_dir or not path.is_file():
        return None
    return path


def get_log_list(request):
    if not LOG_DIR.is_dir():
        return json_response_ensure_ascii({'result': 'success', 'data': []})
    file_list = sorted(path.name for path in LOG_DIR.iterdir() if path.is_file())
    return json_response_ensure_ascii({
        'result': 'success',
        'data': [{'fileName': name} for name in file_list],
    })


def open_log_file(request):
    request_json = json.loads(request.body or b'{}')
    path = _get_log_path(request_json.get('fileName'))
    if path is None:
        return json_response_ensure_ascii({'result': 'error', 'message': 'Log file not found'}, status=404)
    try:
        content = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return json_response_ensure_ascii({'result': 'error', 'message': 'Unable to read log file'}, status=400)
    return json_response_ensure_ascii({'result': 'success', 'data': content})


def clean_log_by_file_name(request):
    request_json = json.loads(request.body or b'{}')
    path = _get_log_path(request_json.get('fileName'))
    if path is None:
        return json_response_ensure_ascii({'result': 'error', 'message': 'Log file not found'}, status=404)
    try:
        path.write_text('', encoding='utf-8')
    except OSError:
        return json_response_ensure_ascii({'result': 'error', 'message': 'Unable to clear log file'}, status=400)
    return json_response_ensure_ascii({'result': 'success', 'data': ''})


def update_log_by_file_name(request):
    return clean_log_by_file_name(request)
