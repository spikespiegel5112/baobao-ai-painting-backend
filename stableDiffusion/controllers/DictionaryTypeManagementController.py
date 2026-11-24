import json

from sqlalchemy import func

from stableDiffusion.database import DictionaryTypes, SessionLocal
from .CommonController import json_response_ensure_ascii


def create(request):
    request_json = json.loads(request.body)
    with SessionLocal() as session:
        session.add(DictionaryTypes(
            title=request_json.get('title'),
            value=request_json.get('value'),
        ))
        session.commit()
    return json_response_ensure_ascii({'result': 'successful'})


def update(request):
    request_json = json.loads(request.body)
    prompt_id = request_json.get('id')
    with SessionLocal() as session:
        data = session.query(DictionaryTypes).filter_by(id=prompt_id).first()
        if data is None:
            return json_response_ensure_ascii({'result': 'error', 'message': 'Dictionary type not found'}, status=404)
        data.title = request_json.get('title')
        data.value = request_json.get('value')
        result_json = data.to_json()
        session.commit()
    return json_response_ensure_ascii({'result': 'successful', 'data': result_json})


def delete(request):
    request_json = json.loads(request.body)
    prompt_id = request_json.get('id')
    with SessionLocal() as session:
        deleted_count = session.query(DictionaryTypes).filter_by(id=prompt_id).delete()
        session.commit()
    if not deleted_count:
        return json_response_ensure_ascii({'result': 'error', 'message': 'Dictionary type not found'}, status=404)
    return json_response_ensure_ascii({'result': 'successful'})


def get(request):
    request_json = json.loads(request.body or b'{}')
    pagination = request_json.get('pagination')
    with SessionLocal() as session:
        total = session.query(func.count(DictionaryTypes.id)).scalar()
        query = session.query(DictionaryTypes).order_by(DictionaryTypes.id)
        if pagination is not None:
            try:
                page = max(1, int(pagination.get('page', 1)))
                size = min(500, max(1, int(pagination.get('size', 50))))
            except (TypeError, ValueError):
                return json_response_ensure_ascii({'result': 'error', 'message': 'Invalid pagination'}, status=400)
            data = query.offset((page - 1) * size).limit(size).all()
        else:
            data = query.all()
        result = [item.to_json() for item in data]
    return json_response_ensure_ascii({
        'result': 'successful',
        'pagination': {'total': total},
        'data': result,
    })
