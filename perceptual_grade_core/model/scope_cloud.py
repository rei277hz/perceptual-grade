"""Portable Warp-cloud descriptors and CPU rasterisation.

This module contains no Blender or GPU dependency. Host adapters can use the
returned uint8 tiles as their only rendering boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any
import numpy as np

POINT_SIZE = 1.5
MAX_OPACITY = 0.35
TONES = 4
CPU_RASTER_CANDIDATES = 131072

def _descriptor(view: Any) -> dict[str, Any]:
    """Normalize a view descriptor without retaining a UI object."""
    if isinstance(view, dict):
        value = dict(view)
    else:
        value = {name: getattr(view, name) for name in
                 ('key', 'width', 'height', 'rect', 'scale', 'bounds')}
    value['key'] = tuple(value.get('key', ()))
    value['width'] = int(value['width'])
    value['height'] = int(value['height'])
    value['rect'] = tuple(float(v) for v in value['rect'])
    value['scale'] = float(value.get('scale', 1.0))
    value['bounds'] = tuple(tuple(float(v) for v in bounds) for bounds in value['bounds'])
    if value['width'] <= 0 or value['height'] <= 0:
        raise ValueError('Warp cloud target dimensions must be positive')
    if len(value['rect']) != 4 or len(value['bounds']) != 4 or any(len(v) != 4 for v in value['bounds']):
        raise ValueError('Warp cloud descriptors require four-component rect/bounds')
    if (not all(math.isfinite(v) for v in (*value['rect'],value['scale'],*sum(value['bounds'],()))) or
            min(value['rect'][2:])<=0 or value['scale']<=0 or
            any(a>=c or b>=d for a,b,c,d in value['bounds'])):
        raise ValueError('Warp cloud rectangles and bounds must be finite and ordered')
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _tone_value(result: Any, tone: int) -> Any:
    colors = _field(result, '_colors')
    if colors is None:
        colors = _field(result, 'colors')
    if colors is None:
        raise ValueError('Scope result has no analytic tone selections')
    return colors[tone]


def _primary(selection: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    coords = _field(selection, 'bounded_saturation')
    if coords is None:
        from .saturation import bounded_project
        coords = bounded_project(_field(selection, 'after'), _field(selection, 'after_energy'))
    weights = _field(selection, 'opacity_weights', _field(selection, 'weights'))
    energy = _field(selection, 'after_energy', _field(selection, 'energy'))
    if coords is None or weights is None:
        raise ValueError('Scope tone selection has no bounded cloud inputs')
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if energy is None:
        energy = np.zeros(weights.shape, dtype=np.float64)
    else:
        # Rasterization only uses the sign. Extended-range Scalar values need
        # not be representable in either binary32 or binary64 to draw a ring.
        energy = np.where(np.asarray(energy).reshape(-1)<0,-1.,1.)
    if len(coords)!=len(weights) or len(coords)!=len(energy):
        raise ValueError('Cloud appearance and membership sizes differ')
    return coords, weights, energy, _field(selection, 'sample_mass')


def _comparison(result: Any, tone: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None:
    values = _field(result, '_comparisons_selected')
    if values is None:
        values = _field(result, 'comparisons')
    if values is None:
        one = _field(result, '_comparison_selected')
        values = None if one is None else (one,)
    if values is None or len(values) == 0:
        return None
    value = values[tone if tone < len(values) else 0]
    signal = _field(value, 'signal', value)
    coords = _field(signal, 'bounded_saturation')
    if coords is None:
        from .saturation import bounded_project
        coords = bounded_project(_field(signal, 'appearance'), _field(signal, 'energy'))
    weights = _field(value, 'opacity_weights', _field(value, 'weights', _field(signal, 'weights', _field(signal, 'alpha'))))
    energy = _field(signal, 'energy', _field(signal, 'after_energy'))
    if coords is None or weights is None:
        return None
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    energy = (np.zeros(weights.shape,dtype=np.float64) if energy is None else
              np.where(np.asarray(energy).reshape(-1)<0,-1.,1.))
    if len(coords)!=len(weights) or len(coords)!=len(energy):
        raise ValueError('Cloud appearance and membership sizes differ')
    return coords, weights, energy, _field(value, 'sample_mass', _field(signal, 'sample_mass'))


def _raster_layer(coords: np.ndarray, weights: np.ndarray, energy: np.ndarray,
                  view: dict[str, Any], tone=0, checkpoint=None, *, mass=None) -> np.ndarray:
    """Rasterize bounded candidate batches with the point shader's coverage.

    Source-over coverage is ``1 - product(1 - alpha)``. Accumulating its log
    avoids a Python loop and temporary pixel grid for every source point,
    while retaining double-size hollow marks for the negative branch.
    """
    width, height = view['width'], view['height']
    rx, ry, rw, rh = view['rect']
    xmin, ymin, xmax, ymax = view['bounds'][tone]
    if not (rw > 0 and rh > 0 and xmax > xmin and ymax > ymin):
        return np.zeros((height, width), dtype=np.float32)
    scale = max(float(view['scale']), 1e-6)
    base_radius = (POINT_SIZE / 2.0) * scale
    log_transmission = np.zeros((height, width), dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    mass = np.ones(weights.shape,dtype=np.float64) if mass is None else np.asarray(mass,dtype=np.float64).reshape(weights.shape)
    energy = np.asarray(energy)
    finite = (np.isfinite(coords).all(axis=-1) & np.isfinite(weights) &
              np.isfinite(energy) & np.isfinite(mass) & (weights > 0) & (mass > 0))
    # The UI shader maps chart (a,b) to screen (-b,a). Keep that mapping here,
    # with row zero at the GL bottom so the transfer has no hidden y flip.
    budget = CPU_RASTER_CANDIDATES
    for negative in (False, True):
        ids = np.flatnonzero(finite & ((energy < 0) == negative))
        radius = base_radius * (2.0 if negative else 1.0)
        reach = (max(width, height) if radius >= max(width, height) else math.ceil(radius))
        footprint = min(width, 2*reach+2) * min(height, 2*reach+2)
        step = min(4096, max(1, budget // min(footprint, budget)))
        for start in range(0, len(ids), step):
            if checkpoint is not None: checkpoint()
            part = ids[start:start+step]
            a, b = coords[part].astype(np.float64).T
            sx = rx + (-b-xmin)*rw/(xmax-xmin)
            sy = ry + (a-ymin)*rh/(ymax-ymin)
            visible = (np.isfinite(sx) & np.isfinite(sy) &
                       (sx+radius >= 0) & (sy+radius >= 0) &
                       (sx-radius < width) & (sy-radius < height))
            part, sx, sy = part[visible], sx[visible], sy[visible]
            if not len(part): continue
            left = np.clip(np.floor(sx-radius), 0, width-1).astype(np.int64)
            right = np.clip(np.ceil(sx+radius), 0, width-1).astype(np.int64)
            bottom = np.clip(np.floor(sy-radius), 0, height-1).astype(np.int64)
            top = np.clip(np.ceil(sy+radius), 0, height-1).astype(np.int64)
            nx, ny = int(np.max(right-left+1)), int(np.max(top-bottom+1))
            pixel_step = max(1, budget // len(part))
            alpha = (np.clip(weights[part], 0., 1.)*MAX_OPACITY).astype(np.float32)
            log_alpha = mass[part] * np.log1p(-alpha.astype(np.float64))
            # Large UI scales can cover the whole canvas. Split those pixel
            # spans too, keeping every temporary array within the same bound.
            for offset in range(0, nx*ny, pixel_step):
                if checkpoint is not None: checkpoint()
                candidates = np.arange(offset, min(offset+pixel_step, nx*ny))
                xx = left[:,None] + candidates % nx
                yy = bottom[:,None] + candidates // nx
                distance = ((xx+.5-sx[:,None])**2 + (yy+.5-sy[:,None])**2)/(radius*radius)
                inside = ((xx <= right[:,None]) & (yy <= top[:,None]) & (distance <= 1.) &
                          (xx+.5 >= rx) & (xx+.5 < rx+rw) & (yy+.5 >= ry) & (yy+.5 < ry+rh))
                if negative: inside &= distance >= .25
                np.add.at(log_transmission.ravel(), yy[inside]*width+xx[inside],
                          np.broadcast_to(log_alpha[:,None], xx.shape)[inside])
    return (-np.expm1(log_transmission)).astype(np.float32)


def _cpu_render(result: Any, views: tuple[dict[str, Any], ...], checkpoint=None) -> tuple[dict[str, Any], ...]:
    records = []
    for view in views:
        started = time.perf_counter()
        layers = 2 if _field(result, '_comparisons_selected') else 1
        pixels = np.zeros((layers, TONES, view['height'], view['width']), dtype=np.uint8)
        for tone in range(TONES):
            primary = _primary(_tone_value(result, tone))
            cloud = _raster_layer(*primary[:3], view, tone, checkpoint, mass=primary[3])
            pixels[0, tone] = np.rint(np.clip(cloud, 0., 1.) * 255.).astype(np.uint8)
            comparison = _comparison(result, tone)
            if comparison is not None:
                cloud = _raster_layer(*comparison[:3], view, tone, checkpoint, mass=comparison[3])
                pixels[1, tone] = np.rint(np.clip(cloud, 0., 1.) * 255.).astype(np.uint8)
        records.append(dict(view=view, pixels=pixels, backend='cpu', timings={
            'cpu_raster_seconds': time.perf_counter() - started,
            'compute_submission_seconds': 0.0,
            'raster_submission_seconds': 0.0,
            'readback_seconds': 0.0,
        }))
    return tuple(records)

