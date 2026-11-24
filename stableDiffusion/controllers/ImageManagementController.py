import base64
import datetime
import json
import os
from pathlib import Path
from io import BytesIO

from PIL import Image
from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from stableDiffusion.database import Images, engine
from .CommonController import json_response_ensure_ascii, generate_file_name

MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024


def _output_path(file_name):
    output_dir = (Path(__file__).resolve().parents[2] / 'output').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError('Invalid image file name')
    path = (output_dir / file_name).resolve()
    if path.parent != output_dir:
        raise ValueError('Invalid image file path')
    return path


def save_image_to_album(request):
    session = None
    try:
        print('=====save_image_to_album=====')
        request_json = json.loads(request.body)
        _num_inference_steps = request_json.get('numInferenceSteps')
        otherSettings = json.dumps(request_json.get('otherSettings'))
        file_name = generate_file_name(_num_inference_steps)

        image_response = save_base64_as_image_file(request, file_name)
        if image_response.status_code >= 400:
            return image_response

        Session = sessionmaker(bind=engine)
        session = Session()
        session.add(
            Images(fileName=file_name,
                   customPositivePrompt=request_json.get('customPositivePrompt'),
                   positivePrompt=request_json.get('positivePrompt'),
                   customNegativePrompt=request_json.get('customNegativePrompt'),
                   negativePrompt=request_json.get('negativePrompt'),
                   width=request_json.get('width'),
                   height=request_json.get('height'),
                   seed=request_json.get('seed'),
                   createdAt=datetime.datetime.now(),
                   otherSettings=otherSettings,
                   userId=request_json.get('userId'),
                   welinkUserId=request_json.get('welinkUserId'),
                   guestUserId=request_json.get('guestUserId'),
                   userComesFrom=request_json.get('userComesFrom'),
                   ))

        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
            'fileName': file_name,
        })
    finally:
        if session is not None:
            session.close()


def save_base64_as_image_file(request, file_name=None):
    print('=====save_base64_as_image_file=====')
    # 从POST请求中获取Base64数据
    request_byte = request.body
    request_json = json.loads(request_byte)

    try:
        if request_json.get('base64Data') is None:
            raise ValueError('base64Data is required')
        if not file_name:
            file_name = generate_file_name(request_json.get('numInferenceSteps'))
        encoded_image = request_json['base64Data']
        if isinstance(encoded_image, str) and encoded_image.startswith('data:image/'):
            header, separator, encoded_image = encoded_image.partition(',')
            if not separator or header.split(';', 1)[0].split('/', 1)[1] not in {'jpeg', 'jpg', 'png', 'gif'}:
                raise ValueError('Unsupported image format')
        image_data = base64.b64decode(encoded_image, validate=True)
        if len(image_data) > MAX_IMAGE_UPLOAD_BYTES:
            raise ValueError('Image exceeds the 20 MB limit')
        with Image.open(BytesIO(image_data)) as source_image:
            source_image.verify()
        with Image.open(BytesIO(image_data)) as source_image:
            if source_image.width * source_image.height > 40_000_000:
                raise ValueError('Image dimensions exceed the 40 megapixel limit')
            image = source_image.convert('RGB')
            image.save(_output_path(file_name), 'JPEG')
        return json_response_ensure_ascii({
            'result': 'successful',
            'fileName': file_name
        })
    except (ValueError, base64.binascii.Error, OSError, Image.DecompressionBombError) as error:
        return json_response_ensure_ascii({
            'result': 'failed',
            'message': str(error),
        }, status=400)


def transform_pil_image_to_base64(pil_image_data):
    buffered = BytesIO()
    pil_image_data.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    image_b64 = (f"data:image/jpeg;base64,{img_str}")

    response_data = image_b64

    return response_data


def get(request):
    session = None
    try:
        request_json = json.loads(request.body)

        pagination = request_json.get('pagination')
        comesFrom = request_json.get('comesFrom')
        welinkUserId = request_json.get('welinkUserId')
        guestUserId = request_json.get('guestUserId')
        is_public = request_json.get('isPublic')
        user_id = request_json.get('userId')
        if pagination is not None:
            page = pagination.get('page')
            size = pagination.get('size')
            offset = (page - 1) * size

        Session = sessionmaker(bind=engine)
        session = Session()
        if comesFrom is None:
            total = session.query(func.count(Images.id)).scalar()
        if is_public is True:
            data = session.query(Images).filter_by(isPublic=is_public)
        else:
            if user_id is not None:
                data = session.query(Images).filter_by(userId=user_id)
            else:
                data = session.query(Images)

        data = data.order_by(desc(Images.createdAt))
        total = data.count()

        if pagination is not None:
            data = data.offset(offset).limit(size)
        else:
            data = data.all()
        result = []
        for item in data:
            result.append(item.to_json())

        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
            'pagination': {
                'total': total
            },
            'data': result
        })
    finally:
        if session is not None:
            session.close()


def create(request):
    session = None
    try:
        request_json = json.loads(request.body)
        Session = sessionmaker(bind=engine)
        session = Session()

        session.add(
            Images(
                fileName=request_json.get('fileName'),
                customPositivePrompt=request_json.get('customPositivePrompt'),
                positivePrompt=request_json.get('positivePrompt'),
                customNegativePrompt=request_json.get('customNegativePrompt'),
                negativePrompt=request_json.get('negativePrompt'),
                width=request_json.get('width'),
                height=request_json.get('height'),
                seed=request_json.get('height'),
                otherSettings=request_json.get('otherSettings'),
                welinkUserId=request_json.get('welinkUserId'),
                guestUserId=request_json.get('guestUserId'),

                createdAt=datetime.datetime.now()
            )

        )

        session.commit()
        session.close()

        return json_response_ensure_ascii({
            'result': 'successful'
        })
    finally:
        if session is not None:
            session.close()


def set_multiple_user_id(request):
    session = None
    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        data = session.query(Images).all()
        for item in data:
            item.userComesFrom = 'welink'

        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
        })
    finally:
        if session is not None:
            session.close()


def set_image_to_public(request):
    session = None
    try:
        request_json = json.loads(request.body)
        image_id = request_json.get('id')
        is_public = request_json.get('isPublic')
        Session = sessionmaker(bind=engine)
        session = Session()

        data = session.query(Images).filter_by(id=image_id).first()
        data.isPublic = is_public

        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
            'id': image_id
        })
    finally:
        if session is not None:
            session.close()


def update(request):
    session = None
    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        request_json = json.loads(request.body)
        prompt_id = request_json.get('id')

        data = session.query(Images).filter_by(id=prompt_id).first()
        data.fileName = request_json.get('fileName')
        data.customPositivePrompt = request_json.get('customPositivePrompt')
        data.positivePrompt = request_json.get('positivePrompt')
        data.customNegativePrompt = request_json.get('customNegativePrompt')
        data.negativePrompt = request_json.get('negativePrompt')
        data.width = request_json.get('width')
        data.height = request_json.get('height')
        data.seed = request_json.get('seed')
        data.otherSettings = request_json.get('otherSettings')
        data.welinkUserId = request_json.get('welinkUserId')
        data.welinkUserId = request_json.get('welinkUserId')
        result_json = data.to_json()

        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
            'data': result_json
        })
    finally:
        if session is not None:
            session.close()


def delete(request):
    session = None
    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        request_json = json.loads(request.body)
        for item in request_json:
            data = session.query(Images).filter_by(id=item)
            for item2 in data:
                file_name = item2.to_json()
                print(file_name['fileName'])
                try:
                    file_path = _output_path(file_name['fileName'])
                except (TypeError, ValueError):
                    continue
                if file_path.exists():
                    file_path.unlink()
                data.delete()

            print('已删除数据的数据量为:', data)
        session.commit()
        session.close()
        return json_response_ensure_ascii({
            'result': 'successful',
        })
    finally:
        if session is not None:
            session.close()
