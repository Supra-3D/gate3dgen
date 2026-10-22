import base64
import io

import pytest
from PIL import Image

from gate3dgen.data.quality_audit.common import QWEN_RUNNER
from gate3dgen.data.quality_audit.qwen_runner import prepare_messages


def test_default_runner_is_packaged():
    assert QWEN_RUNNER.is_file()


def test_embedded_images_preserve_message_order():
    buffer = io.BytesIO()
    Image.new('RGB', (4, 5), (12, 34, 56)).save(buffer, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    messages, images = prepare_messages([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Source'},
        {'type': 'image_url', 'image_url': {'url': url}},
    ]}])
    assert messages[0]['content'] == [{'type': 'text', 'text': 'Source'}, {'type': 'image'}]
    assert images[0].size == (4, 5)
    assert images[0].getpixel((0, 0)) == (12, 34, 56)


def test_local_runner_rejects_remote_images():
    with pytest.raises(ValueError, match='embedded'):
        prepare_messages([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}},
        ]}])
