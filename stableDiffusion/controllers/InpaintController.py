import json
import platform
import time
import base64
import io
import numpy as np

import langid
from PIL import Image, ImageFilter
from django.conf import settings
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    DiffusionPipeline
)
from torch import torch
from torchvision import transforms

from .CommonController import get_repo_id, get_device_name, json_response_ensure_ascii, torch_empty_cache
from .ImageManagementController import transform_pil_image_to_base64
from .MBartController import _translator

is_test_mode = False
execution_duration = 5
cooling_down_duration = 10
temp_file_name = ''
base64_data = None
progress = 0
on_progress = False
cancelling_flag = False
on_cooling_down = False
cooling_down_deadline_timestamp = None
cooling_down_count_down = None
initialized_flag = False
generating_base_index = None
generating_base_steps = None
generating_refiner_index = None
generating_refiner_steps = None
on_progress_session_id = None
last_session_id = None
used_seed = None
temporary_latent_base64 = None
num_inference_steps = None
current_timestep_base = None
current_timestep_refiner = None
current_generating_user_id = None
current_generating_user_comes_from = None
next_image_data_name = None
is_same_user = None
base_repo_id_data = None
pytorch_device = ''


def handle_inpaint(request):
    global on_progress_session_id
    request_bytes = request.body
    request_json = json.loads(request_bytes)
    params = {
        'positive_prompt': request_json.get('positivePrompt'),
        'custom_positive_prompt': request_json.get('customPositivePrompt'),
        'negative_prompt': request_json.get('negativePrompt'),
        'custom_negative_prompt': request_json.get('customNegativePrompt'),
        'num_inference_steps': request_json.get('numInferenceSteps'),
        'width': request_json.get('width'),
        'height': request_json.get('height'),

        'strength': request_json.get('strength'),
        'guidance_scale': request_json.get('guidanceScale'),
        'denoising_end': request_json.get('denoisingEnd'),
        'denoising_start': request_json.get('denoisingStart'),

        'session_id': request_json.get('sessionId'),
        'num_inference_steps_refiner': request_json.get('numInferenceStepsRefiner'),
        'seed': request_json.get('seed'),
        'useFp32': request_json.get('useFp32'),
        'useLcmLora': request_json.get('useLcmLora'),

        'enable_attention_slicing': request_json.get('enableAttentionSlicing'),
        'enable_vae_slicing': request_json.get('enableVaeSlicing'),
        'enable_vae_tiling': request_json.get('enableVaeTiling'),
        'enable_xformers_memory_efficient_attention': request_json.get(
            'enableXformersMemoryEfficientAttention'),

        # welink_user_id : request_json.get('welinkUserId')
        'current_generating_user_comes_from': request_json.get('userComesFrom'),
        'current_generating_user_id': request_json.get('userId'),

        'version_of_stable_diffusion': request_json.get('versionOfStableDiffusion'),

        'image': request_json.get('image'),
        'mask': request_json.get('mask'),
    }
    return inpaint(params)


def inpaint(params):
    global base_repo_id_data
    global pytorch_device
    global is_test_mode
    global on_progress
    global on_cooling_down
    global cooling_down_deadline_timestamp
    global cancelling_flag
    global on_progress_session_id
    global last_session_id
    global current_generating_user_id
    global current_generating_user_comes_from
    global used_seed
    global base64_data
    global cooling_down_count_down
    global cooling_down_duration
    global next_image_data_name
    global is_same_user
    global generating_base_index
    global generating_base_steps
    global generating_refiner_index
    global generating_refiner_steps

    last_session_id = None
    seed = params['seed']
    try:

        g = torch.Generator(device=get_device_name())
        if seed is not None:
            # print('=====seed=====')
            # print(seed)
            seed = int(seed)
            g.manual_seed(seed)
        else:
            seed = g.seed()

        if on_progress is True:
            return json_response_ensure_ascii({
                'result': 'success',
                'code': 10001,
                'message': '有任务在执行'
            })
        else:
            on_progress_session_id = params['session_id']

            current_generating_user_comes_from = params.get(
                'current_generating_user_comes_from', None)
            current_generating_user_id = params.get('current_generating_user_id', None)

        used_seed = seed
        on_progress = True
        on_cooling_down = False

        use_refiner = params['num_inference_steps_refiner'] is not None
        base_repo_id_data = get_repo_id('stable-diffusion-xl-base-1.0')
        base_repo_id = base_repo_id_data.envPath + base_repo_id_data.repoName
        dtype = torch.float16

        print('=====seed=====')
        print(seed)
        print('=====custom_positive_prompt=====')
        print(params.get('custom_positive_prompt'))
        print('=====custom_negative_prompt=====')
        print(params.get('custom_negative_prompt'))
        custom_positive_prompt = params.get('custom_positive_prompt') or ''
        custom_negative_prompt = params.get('custom_negative_prompt') or ''

        print('=====custom_positive_prompt=====')
        print(custom_positive_prompt)
        print('=====custom_negative_prompt=====')
        print(custom_negative_prompt)

        if custom_positive_prompt and langid.classify(custom_positive_prompt)[0] != 'en':
            translated_prompt = _translator(custom_positive_prompt)
            custom_positive_prompt = translated_prompt[0]

        if custom_negative_prompt and langid.classify(custom_negative_prompt)[0] != 'en':
            translated_prompt = _translator(custom_negative_prompt)
            custom_negative_prompt = translated_prompt[0]
        # =========================
        # 1. 加载 SDXL Base
        # =========================

        base_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base_repo_id,
            torch_dtype=dtype,
            variant="fp16",
            low_cpu_mem_usage=False,
        )

        base_pipe.load_lora_weights("/Users/zaoyingouyuandibaozha/Documents/model/xpenis-2.63.safetensors")
        base_pipe.fuse_lora(lora_scale=0.7)  # 或 set_adapters

        base_pipe = base_pipe.to(get_device_name())

        # =========================
        # 3. 加载原图
        # =========================
        print('=====request.image=====')

        image_file = params['image']
        print("===== image_file =====")
        print(image_file)
        print("image_file type:", type(image_file))

        # print(request.image)
        # upload2 returns a browser URL such as /media/uploads/name.png.
        # Resolve it to MEDIA_ROOT before PIL interprets it as a local path.
        if isinstance(image_file, str) and image_file.startswith(settings.MEDIA_URL):
            relative_path = image_file[len(settings.MEDIA_URL):].lstrip('/\\')
            image_path = (settings.MEDIA_ROOT / relative_path).resolve()
            media_root = settings.MEDIA_ROOT.resolve()
            if not image_path.is_relative_to(media_root):
                raise ValueError('Invalid media image path')
            image_file = image_path

        image = Image.open(image_file).convert("RGB")
        width, height = image.size

        # 原图：Base 生成前，将最长边等比例缩放到 1024，
        # 同时保证宽高都是 8 的倍数。
        max_size = 1024
        scale = min(max_size / width, max_size / height, 1.0)

        scaled_width = int(round(width * scale))
        scaled_height = int(round(height * scale))

        # SDXL latent / VAE 要求尺寸最好对齐到 8 的倍数。
        new_width = max(8, (scaled_width // 8) * 8)
        new_height = max(8, (scaled_height // 8) * 8)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        print(
            "image resize:",
            f"{width}x{height} -> {new_width}x{new_height}",
        )

        # mask：base64 -> 灰度 -> 与缩放后的图同尺寸
        mask_raw = params['mask']
        if isinstance(mask_raw, str) and ',' in mask_raw:
            mask_raw = mask_raw.split(',', 1)[1]  # 去掉 data:image/...;base64,
        mask_bytes = base64.b64decode(mask_raw)
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

        print('=====mask=====')
        print('mask size', mask.size, 'image size', image.size)
        print('mask extrema', mask.getextrema())
        print('strength', params['strength'], 'steps', params['num_inference_steps'])
        print('prompt', custom_positive_prompt)
        import numpy as np
        arr = np.array(mask)
        white_ratio = (arr > 127).mean()
        print('white_ratio', white_ratio)  # 理想大概 > 0.02；若 < 0.005 选区太小

        # 1) mask 二值 + 轻羽化（仍须是目标外形，不是整屏方框）
        mask_bin = mask.point(lambda p: 255 if p > 127 else 0)
        ys, xs = np.where(np.array(mask_bin) > 127)
        if len(xs) == 0:
            raise ValueError("empty mask")
        pad = 64  # 外扩，给变长留空
        x0, x1 = max(0, xs.min() - pad), min(image.width, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(image.height, ys.max() + pad)
        # 对齐 8
        x0, y0 = x0 // 8 * 8, y0 // 8 * 8
        x1, y1 = ((x1 + 7) // 8) * 8, ((y1 + 7) // 8) * 8
        crop_img = image.crop((x0, y0, x1, y1))
        crop_mask = mask_bin.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(8))
        # crop_img 本身直接作为模型输入，不再做无实际修改的数组拷贝。
        target = 1024
        cw, ch = crop_img.size
        canvas = Image.new("RGB", (target, target), (127, 127, 127))
        mask_canvas = Image.new("L", (target, target), 0)
        # 居中贴上
        ox, oy = (target - cw) // 2 // 8 * 8, (target - ch) // 2 // 8 * 8
        canvas.paste(crop_img, (ox, oy))
        mask_canvas.paste(crop_mask, (ox, oy))

        # =========================
        # 4. Prompt
        # =========================

        positive_prompt = custom_positive_prompt

        negative_prompt = custom_negative_prompt

        # =========================
        # 5. Base img2img
        # =========================

        print('=========params=========')
        # print(params)
        print({k: v for k, v in params.items() if k != 'mask'})
        pil_image_data = base_pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=canvas,
            mask_image=mask_canvas,
            strength=params['strength'],
            guidance_scale=params['guidance_scale'],
            num_inference_steps=params['num_inference_steps'],
            denoising_end=params['denoising_end'],
            output_type="latent" if use_refiner is True else 'pil',
            width=target,
            height=target,
            generator=g,
            callback_on_step_end=report_base_progress,
        ).images[0]



        # =========================
        # 2. 加载 SDXL Refiner
        # =========================
        # =========================
        # 6. Refiner
        # =========================
        if use_refiner is True:

            refiner_repo_id_data = get_repo_id('stable-diffusion-xl-refiner-1.0')
            refiner_repo_id = refiner_repo_id_data.envPath + refiner_repo_id_data.repoName

            refiner_pipe = DiffusionPipeline.from_pretrained(
                refiner_repo_id,
                text_encoder_2=base_pipe.text_encoder_2,
                vae=base_pipe.vae,
                torch_dtype=dtype,
                variant="fp16",
                low_cpu_mem_usage=False,
            )

            refiner_pipe = refiner_pipe.to(get_device_name())
            pil_image_data = refiner_pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=pil_image_data,
                num_inference_steps=params['num_inference_steps_refiner'],
                denoising_start=params['denoising_start'],
            ).images[0]

        # =========================
        # 7. 保存
        # =========================

        # pil_image_data.save("output.png")
        generated_crop = pil_image_data.crop((ox, oy, ox + cw, oy + ch))

        # 只将 mask 区域的生成结果合成回原图。
        # mask 为黑色的区域严格保留原图，灰度区域进行软融合。
        original_crop = image.crop((x0, y0, x1, y1))
        final_crop = Image.composite(
            generated_crop,
            original_crop,
            crop_mask,
        )

        image.paste(final_crop, (x0, y0))
        pil_image_data = image
        generating_base_index = None
        generating_refiner_index = None
        current_timestep_base = None
        current_timestep_refiner = None
        base64_data = transform_pil_image_to_base64(pil_image_data)

        print("result:", pil_image_data.size)
        json_response = json_response_ensure_ascii({
            'code': 10000,
            'data': base64_data,
            'sessionId': on_progress_session_id,
            'seed': used_seed,
            'userId': current_generating_user_id,
            'userComesFrom': current_generating_user_comes_from,
            'message': '图片生成成功'
        })
        torch_empty_cache()
        if base64_data is not None:
            print('Generation successful...')
        temporary_latent_base64 = None
        if platform.system() != 'Darwin':
            print(torch.cuda.memory_summary())

    except Exception as error:
        print('=====inpaint_error=====')
        print(error)
        print("Torch version:", torch.__version__)
        print("Is CUDA available?", torch.cuda.is_available())
        generating_base_index = None
        generating_refiner_index = None
        current_timestep_base = None
        current_timestep_refiner = None
        on_progress = False
        temporary_latent_base64 = None
        current_generating_user_id = None
        current_generating_user_comes_from = None

        on_progress = False
        cancelling_flag = False
        on_progress_session_id = None

        json_response = json_response_ensure_ascii({
            "error": error
        }, status=400)
        torch_empty_cache()
    finally:
        print('finally!!!!!!!!!!!!!!!!!!!!!!!!!')

    if cancelling_flag is True:
        json_response = json_response_ensure_ascii({
            'code': 10002,
            'sessionId': on_progress_session_id,
            'userId': current_generating_user_id,
            'userComesFrom': current_generating_user_comes_from,
            'message': '生成任务取消'
        })

    on_progress = False
    cancelling_flag = False
    last_session_id = on_progress_session_id
    on_progress_session_id = None
    on_cooling_down = True
    is_same_user = None
    generating_base_index = None
    generating_base_steps = None
    generating_refiner_index = None
    generating_refiner_steps = None
    current_generating_user_id = None
    current_generating_user_comes_from = None
    cooling_down_deadline_timestamp = 0
    cooling_down_deadline_timestamp = cooling_down_duration + time.time()

    return json_response


def report_base_progress(pipe, step_index, timestep, callback_kwargs):
    _vae = pipe.vae
    global generating_base_index
    global current_timestep_base
    global cancelling_flag
    global on_progress
    global initialized_flag
    global temporary_latent_base64
    global base_repo_id_data
    global pytorch_device
    generating_base_index = step_index
    current_timestep_base = int(timestep.cpu().numpy())
    temporary_latent = callback_kwargs.get('latents')
    # 解析latent数据为pil
    # tensor_data = decode_latents(_vae, temporary_latent, pipe)
    # pil_image = tensor_to_pil(tensor_data)
    # temporary_latent_base64 = transform_pil_image_to_base64(pil_image)
    # 不解析latent数据
    # temporary_latent_base64 = transform_latent_to_base64(
    #     temporary_latent[0],
    #     _vae,
    # )
    temporary_latent_base64 = None

    if generating_base_index is not None:
        initialized_flag = True

    if cancelling_flag is True:
        print('cancelled from base!!!')
        cancelling_flag = False
        on_progress = False
        generating_base_index = None

        torch_empty_cache()

        raise Exception("Inference cancelled.")

    return callback_kwargs



def get_generation_status_sdxl_inpaint(request):
    global on_progress
    global cancelling_flag
    global cooling_down_deadline_timestamp
    global cooling_down_count_down
    global generating_base_index
    global generating_base_steps
    global generating_refiner_index
    global generating_refiner_steps
    global temporary_latent_base64
    global current_timestep_base
    global current_timestep_refiner
    global current_generating_user_id
    global on_cooling_down
    global is_same_user

    check_cooling_down()

    request_json = json.loads(request.body)
    app_user_id = request_json.get('appUserId')
    user_id = request_json.get('userId')

    _generating_base_steps = None
    _generating_refiner_steps = None
    if generating_base_steps is not None:
        _generating_base_steps = len(generating_base_steps)
    if generating_refiner_steps is not None:
        _generating_refiner_steps = len(generating_refiner_steps)

    # print('=====current_generating_user_id=====')
    # print(current_generating_user_id)
    # print(user_id)
    # print(user_id == current_generating_user_id)

    if on_progress is True or cancelling_flag is True or on_cooling_down is True:
        is_same_user = user_id == current_generating_user_id
    else:
        is_same_user = None

    if is_same_user is True or is_same_user is None:
        result = {
            'sessionId': on_progress_session_id,
            'onCancelling': cancelling_flag,
            'onProgress': on_progress,
            'onCoolingDown': on_cooling_down,
            'coolingDownCountDown': cooling_down_count_down,
            'generatingIndexBase': generating_base_index,
            'currentTimestepBase': current_timestep_base,
            'generatingIndexRefiner': generating_refiner_index,
            'currentTimestepRefiner': current_timestep_refiner,
            'temporaryLatent': temporary_latent_base64,
            'isSameUser': is_same_user
        }
    else:
        result = {
            'sessionId': on_progress_session_id,
            'onCancelling': cancelling_flag,
            'onProgress': on_progress,
            'onCoolingDown': on_cooling_down,
            'coolingDownCountDown': cooling_down_count_down,
            'isSameUser': False
        }

    response_json = {
        'result': 'success',
    }
    response_json.update(result)

    return json_response_ensure_ascii(response_json)


def check_cooling_down():
    global cooling_down_deadline_timestamp
    global cooling_down_count_down
    global on_cooling_down

    if cooling_down_deadline_timestamp is not None:
        cooling_down_count_down = cooling_down_deadline_timestamp - time.time()
        if cooling_down_count_down < 0:
            cooling_down_deadline_timestamp = None
            cooling_down_count_down = None
            on_cooling_down = False


def transform_latent_to_pil_image(latent_data, vae):
    """将 SDXL latent 通过 VAE 正确解码成 PIL RGB 图片，仅用于进度预览。"""
    if latent_data is None:
        return None

    latent = latent_data
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)

    # Diffusers SDXL VAE 的 latent scaling factor。
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)

    with torch.no_grad():
        latents = latent / scaling_factor
        decoded = vae.decode(latents, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)

    image = decoded[0].detach().float().cpu()
    pil_image = transforms.ToPILImage()(image).convert('RGB')
    return pil_image


def transform_latent_to_base64(latent_data, vae):
    pil_image = transform_latent_to_pil_image(latent_data, vae)
    if pil_image is None:
        return None

    return transform_pil_image_to_base64(pil_image)

