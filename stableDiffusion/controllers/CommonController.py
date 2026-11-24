import base64
import gc
import imghdr
import json
import os
import platform
import time
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from sqlalchemy.orm import sessionmaker
from torch import torch

from stableDiffusion.database import Models
from stableDiffusion.database import Queue
from stableDiffusion.database import engine

on_progress_session_id = None


def generate_file_name(num_inference_steps):
    if num_inference_steps is None:
        num_inference_steps_str = ''
    else:
        num_inference_steps_str = '_' + str(num_inference_steps)
    timestamp_str = str(int(time.time()))
    file_name = timestamp_str + num_inference_steps_str + '.jpg'
    return file_name


def get_architecture_by_os():
    current_os = platform.system()
    print('=====platform======')
    print(current_os)
    if current_os is 'Darwin':
        result = 'mps'
    else:
        result = 'cuda'
    return result


def get_repo_id(repo_name):
    Session = sessionmaker(bind=engine)
    session = Session()
    data = session.query(Models).filter_by(repoName=repo_name)
    session.commit()
    session.close()
    if data.first():
        return data.first()
    else:
        return None


def get_pytorch_device():
    result = 'cuda'
    if platform.system() == 'Darwin':
        result = 'mps'
    return result


def json_response_ensure_ascii(data, status=None):
    if status is None:
        status = 200
    return JsonResponse(data, status=status, json_dumps_params={'ensure_ascii':
                                                                    False})


def get_device_name():
    result = 'cuda'
    # 针对Apple silicon芯片
    if platform.system() == 'Darwin':
        result = 'mps'
    return result


def tensor_to_pil(tensor):
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    image = tensor.numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


def base64_to_pil(base64_str):
    """
      将 base64 字符串转为 RGB 格式的 PIL.Image 对象
      """
    # 如果是 data URL（带前缀），先去掉前缀
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    # 解码 base64 字符串为二进制图像数据
    image_data = base64.b64decode(base64_str)

    # 用 BytesIO 包装为内存流，再用 PIL 打开
    image = Image.open(BytesIO(image_data)).convert('RGB')

    return image


def handle_add_queue(request):
    request_bytes = request.body
    request_json = json.loads(request_bytes)

    params = {
        'sessionId': request_json.get('sessionId'),
        'positivePrompt': request_json.get('positivePrompt'),
        'customPositivePrompt': request_json.get('customPositivePrompt'),
        'negativePrompt': request_json.get('negativePrompt'),
        'customNegativePrompt': request_json.get('customNegativePrompt'),
        'numInferenceSteps': request_json.get('numInferenceSteps'),
        'width': request_json.get('width'),
        'height': request_json.get('height'),
        'seed': request_json.get('seed'),
        'otherSettings': request_json.get('otherSettings'),

        'welinkUserId': request_json.get('welinkUserId'),
        'guestUserId': request_json.get('guestUserId'),
        'userId': request_json.get('userId'),
        'userComesFrom': request_json.get('userComesFrom'),

        'lifeCycle': 'awaiting'
    }
    return add_queue(params)


def handle_add_queue_inpaint(request):
    request_bytes = request.body
    request_json = json.loads(request_bytes)

    params = {
        'sessionId': request_json.get('sessionId'),
        'positivePrompt': request_json.get('positivePrompt'),
        'customPositivePrompt': request_json.get('customPositivePrompt'),
        'negativePrompt': request_json.get('negativePrompt'),
        'customNegativePrompt': request_json.get('customNegativePrompt'),
        'numInferenceSteps': request_json.get('numInferenceSteps'),
        'width': request_json.get('width'),
        'height': request_json.get('height'),
        'seed': request_json.get('seed'),
        'otherSettings': request_json.get('otherSettings'),

        'welinkUserId': request_json.get('welinkUserId'),
        'guestUserId': request_json.get('guestUserId'),
        'userId': request_json.get('userId'),
        'userComesFrom': request_json.get('userComesFrom'),

        'lifeCycle': 'awaiting'
    }
    return add_queue(params)


def add_queue(params):
    Session = sessionmaker(bind=engine)
    session = Session()
    otherSettings = json.dumps(params['otherSettings'])
    print('=====add_queue=====')
    print(params)
    exist_data_by_user_json = []

    # exist_data = session.query(Queue).filter_by(userId=params['userId']).all()
    session.add(
        Queue(
            sessionId=params['sessionId'],
            positivePrompt=params['positivePrompt'],
            customPositivePrompt=params['customPositivePrompt'],
            negativePrompt=params['negativePrompt'],
            customNegativePrompt=params['customNegativePrompt'],
            numInferenceSteps=params['numInferenceSteps'],
            width=params['width'],
            height=params['height'],
            seed=params['seed'],

            otherSettings=otherSettings,

            welinkUserId=params['welinkUserId'],
            guestUserId=params['guestUserId'],
            userId=params['userId'],

            lifeCycle=params['userId'],

            userComesFrom=params['userComesFrom'],
            createdAt=datetime.now(),
        ))
    message = '队列添加成功'
    code = 10000
    # if len(exist_data) == 0:
    # else:
    #     for item in exist_data:
    #         exist_data_by_user_json.append(item.to_json())
    #     message = '本用户队列中已存在未完成任务'
    #     code = 10001

    exist_data = session.query(Queue).filter_by(userId=params['userId']).all()

    for item in exist_data:
        exist_data_by_user_json.append(item.to_json())

    session.commit()
    session.close()

    return json_response_ensure_ascii({
        'result': 'successful',
        'message': message,
        'existData': exist_data_by_user_json,
        'code': code
    })


def check_queue(user_id):
    global on_progress_session_id
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        data = session.query(Queue)
        all_data = data
        user_data = data.filter_by(userId=user_id)

        all_data_length = len(all_data.all())
        user_data_length = len(user_data.all())

        all_data_list = []
        if all_data_length > 0:
            for item in all_data:
                all_data_list.append(item.to_json())

        user_data_list = []
        if user_data_length > 0:
            for item in user_data:
                user_data_list.append(item.to_json())

        # print('=====user_data=====')
        # print(user_data)

        message = None
        code = None
        data = None
        result = []

        if all_data_length == 0:
            message = '整体队列为空'
            code = 10000

        elif user_data_length == 0:
            message = '本用户队列为空'
            code = 10001

        elif user_data_length > 0:
            print('on_progress_session_id+++++', on_progress_session_id)
            for item in user_data_list:
                result.append({
                    'data': item,
                    'active': item['sessionId'] == on_progress_session_id
                })
            message = '本用户队列有待执行任务'
            code = 10002
            data = result

        return {
            'message': message,
            'code': code,
            'data': data,
            'onProgressSessionId': on_progress_session_id,
        }
    finally:
        session.close()


def handle_check_queue_by_user_id(request):
    request_bytes = request.body
    request_json = json.loads(request_bytes)
    user_id = request_json.get('userId')
    queue_info = check_queue(user_id)
    return json_response_ensure_ascii({
        "message": queue_info.get('message'),
        'code': queue_info.get('code'),
        "data": queue_info.get('data'),
        "onProgressSessionId": queue_info.get('on_progress_session_id'),
    })


def delete_queue(request):
    request_bytes = request.body
    request_json = json.loads(request_bytes)
    session_id_list = request_json.get('sessionId')
    Session = sessionmaker(bind=engine)
    session = Session()
    for item in session_id_list:
        data = session.query(Queue).filter_by(sessionId=item)
        data.delete()
    session.commit()
    session.close()

    if len(session_id_list) > 0:
        message = '删除Session成功'
        code = 10000
    else:
        message = '无现存session'
        code = 10001

    return json_response_ensure_ascii({
        "result": 'Successful',
        "data": session_id_list,
        "message": message,
        'code': code
    })


def upload(request):
    print('=====upload start=====')
    print(request)
    print('=====upload end=====')
    image_file = request.FILES.get('file')  # 获取图片对象
    if not image_file:
        return json_response_ensure_ascii({
            'message': '没有上传图片',
            'data': '',
            'status': 400
        })

    # 读取原始二进制数据
    image_data = image_file.read()
    # 编码成 base64
    base64_str = base64.b64encode(image_data).decode('utf-8')
    # 拼接成前端可用的 data URL（可直接用于 <img src="...">）
    data_url = f'data:{image_file.content_type};base64,{base64_str}'

    # 你可以做各种处理，比如：保存文件、处理图片、保存数据库等等
    print("图片名称：", image_file.name)
    print("图片类型：", image_file.content_type)
    print("图片大小：", image_file.size, "bytes")
    return json_response_ensure_ascii({
        'message': '上传成功222',
        'data': data_url,
        'name': image_file.name,
        'type': image_file.content_type,
        'size': image_file.size,
        'status': 200
    })


def upload2(request):
    image_file = request.FILES.get('file')

    if not image_file:
        return json_response_ensure_ascii({
            'message': '没有上传图片',
            'status': 400
        })

    # 检查文件类型
    file_type = imghdr.what(image_file)
    if file_type not in ['jpeg', 'png', 'gif', 'webp']:
        return json_response_ensure_ascii({
            'message': '不支持的图片类型',
            'status': 400
        })

    upload_dir = settings.MEDIA_ROOT / 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    file_ext = os.path.splitext(image_file.name)[1].lower() or f'.{file_type}'
    file_name = f"{timezone.now():%Y%m%d%H%M%S%f}{file_ext}"
    file_path = upload_dir / file_name
    with open(file_path, 'wb+') as destination:
        for chunk in image_file.chunks():
            destination.write(chunk)
    image_obj = Image.open(image_file).convert("RGB")

    return json_response_ensure_ascii({
        'message': '上传成功',
        'filePath': f'{settings.MEDIA_URL}uploads/{file_name}',
        'name': image_file.name,
        'type': file_type,
        'size': image_file.size,
        'width': image_obj.width,
        'height': image_obj.height,
        'status': 200
    })


def model_to_dict(model):
    return {
        column.name: getattr(model, column.name)
        for column in model.__table__.columns
    }


def torch_empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
