"""Small dependency-free PNG decoder for bundled RGBA8 extension assets."""

from ..integration.components import Component

def create(_application):
    tr = _application.component('localization.i18n').tr

    from pathlib import Path
    import struct
    import zlib

    import numpy as np


    _SIGNATURE = b"\x89PNG\r\n\x1a\n"


    def _paeth(a, b, c):
        estimate = a + b - c
        pa = abs(estimate - a)
        pb = abs(estimate - b)
        pc = abs(estimate - c)
        return a if pa <= pb and pa <= pc else b if pb <= pc else c


    def load_rgba8(path):
        """Return an unfiltered ``height x width x 4`` uint8 RGBA array."""
        payload = Path(path).read_bytes()
        if not payload.startswith(_SIGNATURE):
            raise ValueError(tr('error.png_signature'))
        offset = len(_SIGNATURE)
        width = height = bit_depth = color_type = interlace = None
        compressed = bytearray()
        while offset < len(payload):
            length = struct.unpack_from(">I", payload, offset)[0]
            kind = payload[offset + 4:offset + 8]
            chunk = payload[offset + 8:offset + 8 + length]
            offset += length + 12
            if kind == b"IHDR":
                width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
            elif kind == b"IDAT":
                compressed.extend(chunk)
            elif kind == b"IEND":
                break
        if (width is None or height is None or bit_depth != 8 or color_type != 6
                or interlace != 0):
            raise ValueError(tr('error.png_format'))
        stride = width * 4
        encoded = zlib.decompress(compressed)
        expected = height * (stride + 1)
        if len(encoded) != expected:
            raise ValueError(tr('error.png_scanline'))
        rows = []
        previous = bytearray(stride)
        cursor = 0
        for _ in range(height):
            filter_type = encoded[cursor]
            cursor += 1
            current = bytearray(encoded[cursor:cursor + stride])
            cursor += stride
            for index, value in enumerate(current):
                left = current[index - 4] if index >= 4 else 0
                above = previous[index]
                upper_left = previous[index - 4] if index >= 4 else 0
                if filter_type == 1:
                    current[index] = (value + left) & 255
                elif filter_type == 2:
                    current[index] = (value + above) & 255
                elif filter_type == 3:
                    current[index] = (value + ((left + above) // 2)) & 255
                elif filter_type == 4:
                    current[index] = (value + _paeth(left, above, upper_left)) & 255
                elif filter_type != 0:
                    raise ValueError(tr('error.png_filter'))
            rows.append(current)
            previous = current
        return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, 4)

    def _set_export__SIGNATURE(value):
        nonlocal _SIGNATURE
        _SIGNATURE = value

    def _set_export__paeth(value):
        nonlocal _paeth
        _paeth = value

    def _set_export_load_rgba8(value):
        nonlocal load_rgba8
        load_rgba8 = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Path': lambda: Path,
        '_SIGNATURE': lambda: _SIGNATURE,
        '_paeth': lambda: _paeth,
        'load_rgba8': lambda: load_rgba8,
        'np': lambda: np,
        'struct': lambda: struct,
        'tr': lambda: tr,
        'zlib': lambda: zlib,
    }, writers={
        '_SIGNATURE': _set_export__SIGNATURE,
    '_paeth': _set_export__paeth, 'load_rgba8': _set_export_load_rgba8})
