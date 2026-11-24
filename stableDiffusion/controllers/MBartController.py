# from diffusers import StableDiffusionPipeline
import json
import threading

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from .CommonController import get_repo_id
from .CommonController import json_response_ensure_ascii


# from database import engine
# from stableDiffusion.models.Images import Images

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _get_model_and_tokenizer(repo_id):
    """Load the large translation model once per worker process."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        with _model_lock:
            if _model is None or _tokenizer is None:
                _tokenizer = MBart50TokenizerFast.from_pretrained(repo_id)
                _model = MBartForConditionalGeneration.from_pretrained(repo_id)
                _model.eval()
    return _model, _tokenizer


def translator(request):
    request_json = json.loads(request.body)
    source = request_json.get('source')
    result = _translator(source)

    print('=====tokenizer=====')
    print(result)

    return json_response_ensure_ascii({
        'result': 'success',
        'data': result[0]
    })


def _translator(source):
    repo_id_data = get_repo_id('mbart-large-50-many-to-many-mmt')
    repo_id = repo_id_data.envPath + repo_id_data.repoName
    print('=====repo_id=====')
    print(repo_id)
    model, tokenizer = _get_model_and_tokenizer(repo_id)

    # translate Arabic to English
    tokenizer.src_lang = "zh_CN"
    encoded_ar = tokenizer(source, return_tensors="pt")
    # Transformers/PyTorch native generation can share worker resources; serialize
    # access to the cached model and avoid building an autograd graph.
    with _model_lock, torch.inference_mode():
        generated_tokens = model.generate(
            **encoded_ar,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return result
