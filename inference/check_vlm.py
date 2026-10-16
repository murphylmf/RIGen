from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8012/v1')
    parser.add_argument('--model', default='internvl3-8b')
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_path = args.image
    if image_path is None:
        image_path = next(iter(sorted((root / 'cases').rglob('condition.png'))), None)
    if image_path is None:
        image_path = next(iter(sorted((root / 'cases').rglob('front.png'))), None)
    if image_path is None:
        raise FileNotFoundError('Pass --image or install the bundled example cases')
    endpoint = args.base_url.rstrip('/')
    response = requests.get(endpoint + '/models', timeout=30)
    response.raise_for_status()
    model_ids = [item['id'] for item in response.json()['data']]
    if args.model not in model_ids:
        raise ValueError(f'Model {args.model!r} is not served; available: {model_ids}')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    content = [{'type': 'text', 'text': 'Describe these views briefly. Return JSON with one string field named description.'}]
    content.extend({'type': 'image_url', 'image_url': {'url': url}} for _ in range(4))
    schema = {'type': 'object', 'properties': {'description': {'type': 'string'}}, 'required': ['description'], 'additionalProperties': False}
    response = requests.post(endpoint + '/chat/completions', json={
        'model': args.model, 'temperature': 0, 'max_tokens': 128,
        'messages': [{'role': 'user', 'content': content}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'vlm_setup_check', 'strict': True, 'schema': schema}},
    }, timeout=180)
    response.raise_for_status()
    choice = response.json()['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise RuntimeError('The VLM response did not complete')
    result = json.loads(choice['message']['content'])
    if set(result) != {'description'} or not isinstance(result['description'], str):
        raise ValueError(f'Unexpected response: {result}')
    print(json.dumps({'model': args.model, 'images': 4, 'structured_output': 'passed', 'response': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
