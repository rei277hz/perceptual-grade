"""Fixed ACES 2 HDR Rec.2020 profile from OCIO 2.5, implemented in NumPy.

This is the legacy browser's canonical transform, not a configurable ACES
output transform. The physical-unit boundary matches ``aces_output.rs``:
XYZ Y=1 enters the inverse directly, without a Photoshop 2.03 multiplier.
All functions accept arrays with a final RGB/XYZ dimension of three.
Profile matrices and HDR lookup arrays are also shared with GPU consumers.
"""
import numpy as np

from ._aces2_tables import HDR_REACH_M, HDR_GAMUT_CUSP, HDR_GAMUT_HUES

AP0_TO_ACESCG = np.array([[1.4514393161456653, -.23651074689374019, -.21492856925192524],
                         [-.07655377339602043, 1.1762296998335731, -.0996759264375522],
                         [.008316148425697719, -.006032449791021028, .9977163013653233]])
ACESCG_TO_AP0 = np.array([[.6954522413574518, .14067869647029416, .16386906217225403],
                         [.04479456337203763, .8596711184564216, .09553431817154036],
                         [-.005525882558113544, .004025210305978659, 1.001500672252135]])
REC2020_TO_XYZ = np.array([[.63695804830129099, .14461690358620841, .16888097516417205],
                           [.26270021201126692, .67799807151887115, .059301716469861938],
                           [4.9941065744660705e-17, .028072693049087431, 1.0609850577107902]])
XYZ_TO_REC2020 = np.array([[1.716651187971268, -.3556707837763925, -.2533662813736599],
                           [-.666684351832489, 1.6164812366349388, .0157685458139111],
                           [.0176398574453108, -.0427706132578085, .9421031212354739]])
RGB_TO_LMS = np.array([[.312590361, .368067265, .113042921],
                       [.108960696, .542771041, .141968831],
                       [.0084437551, .0436396375, .741617143]])
AP0_TO_LMS = np.array([[.445181042, .34964928, -.00112973212],
                       [.123734146, .613643706, .0563228019],
                       [.0117007261, .0280607939, .753939033]])
AAB_FROM_RGB_A = np.array([[20.25881, 10.129405, .506470263],
                           [15480, -16887.2734, 1407.27271], [1720, 1720, -3440]])
AAB_TO_RGB_A = np.array([[.0323680267, 2.07657631e-5, 1.32606210e-5],
                         [.0323680267, -4.10250432e-5, -1.20174373e-5],
                         [.0323680267, -1.01296409e-5, -2.90076074e-4]])
LMS_TO_AP0 = np.array([[2.66705441, -1.52505875, .117925502],
                      [-.535811961, 1.94158089, -.145848125],
                      [-.0214489009, -.0485954471, 1.32996535]])
LMS_TO_REC2020 = np.array([[4.18920946, -2.83307695, -.0962111205],
                           [-.841454685, 2.4402566, -.338880867],
                           [.00181781093, -.111337908, 1.36944115]])
J_MAX = 283.249878
FOCUS_J = 40.816883
SLOPE_GAIN = 1051.56519
MNORM_COS = np.array([28.177105004310604, 40.918782982468606, 19.588056175708772])
MNORM_SIN = np.array([36.435131137728689, -15.832440585136412, 22.842479106426239])


def _rgb_to_jmh(rgb, matrix):
    lms = np.asarray(rgb, dtype=np.float64) @ matrix.T
    power = np.abs(lms)**.42
    aab = (np.sign(lms)*power/(27.1299992+power)) @ AAB_FROM_RGB_A.T
    j = 100*np.maximum(aab[..., 0], 0)**1.13705599
    m = np.hypot(aab[..., 1], aab[..., 2])
    h = np.degrees(np.arctan2(aab[..., 2], aab[..., 1])) % 360
    return np.stack((j, np.where(aab[..., 0] > 0, m, 0),
                     np.where(aab[..., 0] > 0, h, 0)), axis=-1)


def _reach(h):
    h = h % 360
    base = np.floor(h).astype(np.int64)+1
    return HDR_REACH_M[base]+(HDR_REACH_M[base+1]-HDR_REACH_M[base])*(h-np.floor(h))


def _cusp(h):
    # OCIO's search is deliberately bounded around floor(h), including its
    # two sentinel samples. A global searchsorted is not equivalent.
    h = h % 360
    i = np.floor(h).astype(np.int64)+1
    lo, hi = np.maximum(i-1, 0), np.minimum(i+2, 361)
    for _ in range(2):
        active = lo+1 < hi
        above = h > HDR_GAMUT_HUES[i]
        lo = np.where(active & above, i, lo)
        hi = np.where(active & ~above, i, hi)
        i = (lo+hi)//2
    lower, upper = hi-1, hi
    t = np.clip((h-HDR_GAMUT_HUES[lower])/(HDR_GAMUT_HUES[upper]-HDR_GAMUT_HUES[lower]), 0, 1)
    table = HDR_GAMUT_CUSP.reshape(-1, 3)
    return table[lower]+(table[upper]-table[lower])*t[..., None]


def _j_intersect(j, m, focus, slope_gain):
    scaled = m/slope_gain
    a = scaled/focus
    below = j < focus
    b = np.where(below, 1-scaled, -(1+scaled+J_MAX*a))
    c = np.where(below, -j, J_MAX*scaled+j)
    root = np.sqrt(b*b-4*a*c)
    return -2*c/np.where(below, b+root, b-root)


def _gamut(jmh, jx, inverse, reach_boundary=None):
    j, m, h = np.moveaxis(jmh, -1, 0)
    cusp = _cusp(h)
    cj, cm, gamma = np.moveaxis(cusp, -1, 0)
    weight = np.minimum(1.3-cj/J_MAX, 1)
    focus = cj*(1-weight)+FOCUS_J*weight
    threshold_j = cj*.7+J_MAX*.3
    gain = np.where(jx > threshold_j, np.log10((J_MAX-threshold_j)/np.maximum(.0001, J_MAX-jx))**2+1, 1)
    slope_gain = SLOPE_GAIN*gain
    source_j = _j_intersect(j, m, focus, slope_gain)
    base = np.where(source_j < focus, source_j, J_MAX-source_j)
    slope = base*(source_j-focus)/(focus*slope_gain)
    cusp_j = _j_intersect(cj, cm, focus, slope_gain)
    lower = cusp_j*(source_j/cusp_j)**.826446235/(cj/cm-slope)
    upper = cm*(J_MAX-cusp_j)*((J_MAX-source_j)/(J_MAX-cusp_j))**gamma/(slope*cm+J_MAX-cj)
    smoothing = .12*cm
    blend = np.maximum(smoothing-np.abs(lower-upper), 0)/smoothing
    boundary = np.minimum(lower, upper)-blend**3*smoothing/6
    reach = J_MAX*(source_j/J_MAX)**.879464149/(J_MAX/(_reach(h) if reach_boundary is None else reach_boundary)-slope)
    proportion = np.maximum(boundary/reach, .75)
    threshold = proportion*boundary
    reach_offset, gamut_offset = reach-threshold, boundary-threshold
    scale = reach_offset/(reach_offset/gamut_offset-1)
    nd = (m-threshold)/scale
    if inverse:
        remapped = threshold+scale*np.where(nd >= 1, 1, -(nd/(nd-1)))
    else:
        remapped = threshold+scale*nd/(1+nd)
    remapped = np.where((proportion >= 1) | (m <= threshold), m, remapped)
    result = np.stack((source_j+remapped*slope, remapped, h), axis=-1)
    zero = (m <= 0) | (j > J_MAX) | (boundary <= 0)
    return np.where(zero[..., None], np.stack((j, np.zeros_like(j), h), axis=-1), result)


def _tone(j, inverse):
    a = .0323680267*(np.abs(j)*.00999999978)**.879464149
    y = (27.1299992*a/(1-a))**2.3809523809523809
    if inverse:
        z = np.clip(y/.79370057210326195, 0, 10.1325417)
        ht = .5*(z+np.sqrt(z*(.15999999642372131+z)))
        yo = 4.6796021847256952/((10.1729107/ht)**.86956523541917463-1)
        power = np.abs(yo)**.42
    else:
        f = 10.1729107*(y/(y+4.6796021847256952))**1.14999998
        y_ts = np.maximum(f*f/(f+.0399999991), 0)
        power = (.79370057210326195*y_ts)**.42
    return np.sign(j)*100*(power/(27.1299992+power)*30.8946857)**1.13705599


def _toe(x, limit, k1, k2, inverse):
    k2 = np.maximum(k2, .001)
    k1 = np.sqrt(k1*k1+k2*k2)
    k3 = (limit+k1)/(limit+k2)
    if inverse:
        result = (x*x+k1*x)/(k3*(x+k2))
    else:
        value = k3*x-k1
        result = .5*(value+np.sqrt(value*value+4*k2*k3*x))
    return np.where(x > limit, x, result)


def _chroma(jmh, inverse):
    j_in, m_in, h = np.moveaxis(jmh, -1, 0)
    j_out = _tone(j_in, inverse)
    j_ts = j_in if inverse else j_out
    radians = np.radians(h)
    harmonics = radians[..., None]*np.arange(1, 4)
    mnorm = np.cos(harmonics)@MNORM_COS+np.sin(harmonics)@MNORM_SIN+191.63428819274904
    nj = j_ts/J_MAX
    snj = np.maximum(1-nj, 0)
    limit = nj**.879464149*_reach(h)/mnorm
    if inverse:
        m = _toe(m_in/mnorm, limit, nj*10.3199997, snj, True)
        m = limit-_toe(limit-m, limit-.001, snj*.402999997, np.sqrt(nj*nj+.000500000024), True)
        m *= mnorm*(j_ts/j_out)**-.879464149
    else:
        m = m_in*(j_ts/j_in)**.879464149/mnorm
        m = limit-_toe(limit-m, limit-.001, snj*.402999997, np.sqrt(nj*nj+.000500000024), False)
        m = _toe(m, limit, nj*10.3199997, snj, False)*mnorm
    m = np.where((m_in == 0) | ((j_in == 0) & (not inverse)), 0, m)
    return np.stack((j_out, m, h), axis=-1)


def _jmh_to_rgb(jmh, matrix):
    j, m, h = np.moveaxis(jmh, -1, 0)
    radians = np.radians(h)
    aab = np.stack(((j*.00999999978)**.879464149, m*np.cos(radians), m*np.sin(radians)), axis=-1)
    rgb_a = aab@AAB_TO_RGB_A.T
    limited = np.minimum(np.abs(rgb_a), .99000001)
    lms = np.sign(rgb_a)*(27.1299992*limited/(1-limited))**2.38095236
    return lms@matrix.T


def inverse_from_xyz_d65(xyz):
    """Display-referred XYZ-D65 to canonical scene-linear ACEScg."""
    with np.errstate(all='ignore'):
        rgb = np.clip(np.asarray(xyz, dtype=np.float64)@XYZ_TO_REC2020.T, 0, 10)
        jmh = _rgb_to_jmh(rgb, RGB_TO_LMS)
        first = _gamut(jmh, jmh[..., 0], True)
        threshold = _cusp(jmh[..., 2])[..., 0]*.7+J_MAX*.3
        compressed = np.where((jmh[..., 0] > threshold)[..., None], _gamut(jmh, first[..., 0], True), first)
        return np.clip(_jmh_to_rgb(_chroma(compressed, True), LMS_TO_AP0)@AP0_TO_ACESCG.T, 0, 4096)


def forward_to_xyz_d65(acescg):
    """Canonical scene-linear ACEScg to display-referred HDR Rec.2020 XYZ."""
    with np.errstate(all='ignore'):
        jmh = _rgb_to_jmh(np.asarray(acescg, dtype=np.float64)@ACESCG_TO_AP0.T, AP0_TO_LMS)
        chroma = _chroma(jmh, False)
        compressed = _gamut(chroma, chroma[..., 0], False, _reach(jmh[..., 2]))
        return np.clip(_jmh_to_rgb(compressed, LMS_TO_REC2020), 0, 10)@REC2020_TO_XYZ.T
