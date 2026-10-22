#!/usr/bin/env python3
"""Create Blender-3-compatible GLB copies by transcoding EXT_texture_webp.

The source GLB is never modified.  PNG payloads are appended to the existing
binary buffer, texture source references are made core glTF references, and
the now-unused WebP extension declaration is removed.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

JSON_CHUNK = b"JSON"
BIN_CHUNK = b"BIN\x00"


def _pad(value: bytes, fill: bytes) -> bytes:
    return value + fill * ((-len(value)) % 4)


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"truncated GLB: {path}")
    magic, version, declared = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared != len(payload):
        raise ValueError(f"invalid GLB v2 header: {path}")
    cursor = 12
    document: dict[str, Any] | None = None
    binary: bytes | None = None
    while cursor < len(payload):
        if len(payload) - cursor < 8:
            raise ValueError(f"truncated GLB chunk header: {path}")
        length, kind = struct.unpack_from("<I4s", payload, cursor)
        cursor += 8
        chunk = payload[cursor:cursor + length]
        cursor += length
        if len(chunk) != length:
            raise ValueError(f"truncated GLB chunk: {path}")
        if kind == JSON_CHUNK:
            document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
        elif kind == BIN_CHUNK:
            binary = chunk
    if not isinstance(document, dict) or binary is None:
        raise ValueError(f"GLB lacks JSON/BIN chunk: {path}")
    return document, binary


def write_glb(path: Path, document: Mapping[str, Any], binary: bytes) -> None:
    json_bytes = _pad(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), b" ")
    binary = _pad(binary, b"\x00")
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_bytes), JSON_CHUNK)
        + json_bytes
        + struct.pack("<I4s", len(binary), BIN_CHUNK)
        + binary
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def convert(source: Path, destination: Path) -> dict[str, Any]:
    from PIL import Image

    if source.resolve() == destination.resolve():
        raise ValueError("source GLB must remain read-only; choose a different destination")
    document, binary = read_glb(source)
    images = document.get("images") or []
    buffer_views = document.get("bufferViews") or []
    textures = document.get("textures") or []
    if not isinstance(images, list) or not isinstance(buffer_views, list) or not isinstance(textures, list):
        raise ValueError(f"invalid GLB image/texture arrays: {source}")
    converted: dict[int, int] = {}
    output_binary = bytearray(binary)
    for index, image in enumerate(images):
        if not isinstance(image, dict) or image.get("mimeType") != "image/webp":
            continue
        view_index = int(image["bufferView"])
        view = buffer_views[view_index]
        if not isinstance(view, dict) or int(view.get("buffer", 0)) != 0:
            raise ValueError(f"unsupported WebP buffer view {view_index}: {source}")
        offset = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        webp = binary[offset:offset + length]
        with Image.open(io.BytesIO(webp)) as decoded:
            png_buffer = io.BytesIO()
            decoded.save(png_buffer, format="PNG")
            png = png_buffer.getvalue()
        while len(output_binary) % 4:
            output_binary.append(0)
        new_view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": len(output_binary), "byteLength": len(png)})
        output_binary.extend(png)
        images[index] = {**image, "bufferView": new_view_index, "mimeType": "image/png"}
        converted[index] = new_view_index

    for texture in textures:
        if not isinstance(texture, dict):
            continue
        extensions = texture.get("extensions")
        webp_extension = extensions.get("EXT_texture_webp") if isinstance(extensions, dict) else None
        if isinstance(webp_extension, dict):
            source_index = int(webp_extension["source"])
            if source_index not in converted:
                raise ValueError(f"texture references a non-WebP EXT source {source_index}: {source}")
            texture["source"] = source_index
            del extensions["EXT_texture_webp"]
            if not extensions:
                texture.pop("extensions", None)
    for key in ("extensionsUsed", "extensionsRequired"):
        values = document.get(key)
        if isinstance(values, list):
            document[key] = [value for value in values if value != "EXT_texture_webp"]
            if not document[key]:
                document.pop(key, None)
    if document.get("buffers"):
        document["buffers"][0]["byteLength"] = len(output_binary)
    write_glb(destination, document, bytes(output_binary))
    return {"source": str(source), "destination": str(destination), "converted_images": len(converted)}


def blender_compatible(path: Path) -> bool:
    try:
        document, _ = read_glb(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        "EXT_texture_webp" not in (document.get("extensionsUsed") or [])
        and "EXT_texture_webp" not in (document.get("extensionsRequired") or [])
        and all(image.get("mimeType") != "image/webp" for image in (document.get("images") or []))
        and all("EXT_texture_webp" not in (texture.get("extensions") or {})
                for texture in (document.get("textures") or []))
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = convert(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
