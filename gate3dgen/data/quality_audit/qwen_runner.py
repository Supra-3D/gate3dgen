"""Local, offline Qwen3-VL audit worker using the audit JSON-line protocol."""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys


def prepare_messages(messages):
    """Decode embedded audit images; never fetch remote image URLs."""
    from PIL import Image

    converted, images = [], []
    for message in messages:
        content = []
        for item in message['content']:
            if item['type'] == 'text':
                content.append({'type': 'text', 'text': item['text']})
            elif item['type'] == 'image_url':
                url = item['image_url']['url']
                if not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Local audit accepts embedded base64 images only')
                data = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
                with Image.open(io.BytesIO(data)) as image:
                    images.append(image.convert('RGB'))
                content.append({'type': 'image'})
            else:
                raise ValueError(f"Unsupported message content: {item['type']}")
        converted.append({'role': message['role'], 'content': content})
    return converted, images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=2400)
    parser.add_argument('--mode', choices=['vl'], default='vl')
    parser.add_argument('--server', action='store_true', required=True)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--enable_thinking', choices=['false'], default='false')
    parser.add_argument('--local_files_only', action='store_true')
    args = parser.parse_args(argv)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError('Local Qwen3-VL audit requires a CUDA GPU')
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa', device_map={'': 'cuda:0'},
    ).eval()
    print('__QWEN_SERVER_READY__', flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            messages, images = prepare_messages(request['messages'])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer = processor.batch_decode(result[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            response = {'ok': True, 'text': answer}
        except Exception as exc:
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        print('__QWEN_RESPONSE__' + json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
