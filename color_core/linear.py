"""Model-neutral linear color algebra and standard RGB spaces."""
from functools import lru_cache
import math

I3 = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


CAT16 = ((.401288, .650173, -.051461), (-.250268, 1.204414, .045854),
         (-.002079, .048952, .953127))


D65 = (.3127, .3290)


ACES_WHITE = (.32168, .33767)


WHITES = {
    'D65': D65, 'D60': (.32161671, .33761992), 'ACES': ACES_WHITE,
    'D50': (.34567, .35850), 'D55': (.33242, .34743),
    'D75': (.29903, .31488), 'DCI': (.314, .351), 'E': (1/3, 1/3),
}


REC709 = ((.64, .33), (.30, .60), (.15, .06))


BRADFORD = ((.8951, .2664, -.1614), (-.7502, 1.7135, .0367),
            (.0389, -.0685, 1.0296))


SPACES = {
    'Linear Rec.709': (REC709, D65),
    'Linear Rec.2020': (((.708, .292), (.170, .797), (.131, .046)), D65),
    'ACEScg': (((.713, .293), (.165, .830), (.128, .044)), ACES_WHITE),
}


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def mv(m, v):
    return tuple(dot(row, v) for row in m)


def mm(a, b):
    return tuple(tuple(dot(row, col) for col in zip(*b)) for row in a)


def columns(*cols):
    return tuple(tuple(row) for row in zip(*cols))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def inverse(m):
    u, v, w = cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])
    det = dot(m[0], u)
    if not math.isfinite(det) or abs(det) < 1e-14:
        raise ValueError('Singular color basis.')
    return tuple(tuple(x/det for x in row) for row in columns(u, v, w))


ICAT16 = inverse(CAT16)


def xyz(p, Y=1.):
    x, y = p
    if y <= 0 or not all(math.isfinite(v) for v in (x, y, Y)):
        raise ValueError('Invalid xy white/reference.')
    return (x*Y/y, Y, (1-x-y)*Y/y)


def xy(v):
    s = sum(v)
    if abs(s) < 1e-14 or not math.isfinite(s):
        return None
    return (v[0]/s, v[1]/s)


@lru_cache(maxsize=64)
def bradford(source, destination):
    """Full Bradford adaptation of XYZ, with both whites normalized to Y=1."""
    if source == destination:
        return I3
    a, b = mv(BRADFORD, xyz(source)), mv(BRADFORD, xyz(destination))
    return mm(inverse(BRADFORD), tuple(tuple(x*b[i]/a[i] for x in row)
                                     for i, row in enumerate(BRADFORD)))


def cat16(source, destination):
    """Full CAT16 adaptation between xy whites normalized to Y=1.

    Both whites must have positive finite CAT16 cone responses. No source-white
    projection is applied; applications may constrain their controls separately.
    """
    a, b = mv(CAT16, xyz(source)), mv(CAT16, xyz(destination))
    if any(not math.isfinite(v) or v<=0 for v in (*a,*b)):
        raise ValueError('CAT16 adaptation requires positive finite cone gains')
    gains=tuple(y/x for x,y in zip(a,b))
    if any(not math.isfinite(v) or v<=0 for v in gains):
        raise ValueError('CAT16 adaptation requires positive finite cone gains')
    delta=mm(ICAT16,tuple(tuple(x*(gains[i]-1) for x in row) for i,row in enumerate(CAT16)))
    return tuple(tuple(I3[i][j]+delta[i][j] for j in range(3)) for i in range(3))


def rebase(matrix, source, destination):
    """Change a correction's XYZ white context without changing its action."""
    if source == destination or all(matrix[i][j] == (matrix[0][0] if i == j else 0.)
                                    for i in range(3) for j in range(3)):
        return matrix
    cat = bradford(source, destination)
    return mm(cat, mm(matrix, inverse(cat)))


@lru_cache(maxsize=16)
def rgb_to_xyz(space):
    primaries, wp = SPACES[space]
    p = columns(*(xyz(value) for value in primaries))
    scale = mv(inverse(p), xyz(wp))
    return tuple(tuple(row[index] * scale[index] for index in range(3)) for row in p)


@lru_cache(maxsize=16)
def xyz_to_rgb(space):
    return inverse(rgb_to_xyz(space))


def convert_rgb(value, source, destination):
    """Convert scene-linear RGB using Bradford white adaptation, without clipping."""
    if source == destination:
        return tuple(float(component) for component in value)
    return mv(mm(xyz_to_rgb(destination),
                 mm(bradford(SPACES[source][1], SPACES[destination][1]), rgb_to_xyz(source))), value)
