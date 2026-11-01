"""Tonal J/C controls and the seven-handle hue/chroma/Adaptance web."""
from dataclasses import dataclass, make_dataclass, replace
from functools import lru_cache, cached_property
import numpy as np
from . import tone as tm
from ...color_core.scaled import Scalar as S, total, dot, norm
from ...color_core.modcam16 import (HK, State, Appearance, hk_energy, requested_energy, achromatic_response,
                        Response, INVERSE, eccentricity, uncompress, GAMMA)
from .tonal_web import WebSettings, TonalWeb, cardinal_values, smoother, rec709_contour
from .selective_adaptance import SelectiveAdaptance, GuardedWarpDisplay
from .saturation import project, bounded_project, bounded_unproject

TONES=('Shadows','Midtones','White','Highlights')
CHROMA_CROSSOVER=.4
IDS='SMWH'
CONTROLS=[]
def add(name,default,lo,hi,precision,dctl):
    CONTROLS.append((name,name,default,lo,hi,precision,dctl))
for axis in 'ab':
    for i,tid in enumerate(IDS):
        add(f'adaptance_{axis}_{i}',0.,-1.e6,1.e6,2,f'tone{tid}Adaptance{axis.upper()}')
add('adaptance_half_saturation',.25,.05,1,2,'adaptanceHalfSaturation')
LIMITS_J={'low':(30,50,100,200),'high':(100,200,400,700),'a':(20,20,30,50),'b':(20,20,30,50)}
LIMITS_C={'low':(4,)*4,'high':(16,)*4,'a':(4,2,1,1),'b':(4,2,1,1)}
DRIVERS=(('low','Cₗ','Low'),('high','Cₕ','High'),('a','a','a'),('b','b','b'))
for family,target,limits,fit in [('j','J',LIMITS_J,'lightnessFit'),('c','C',LIMITS_C,'chromaFit')]:
    for driver,label,dctl in DRIVERS:
        for i,(tid,limit) in enumerate(zip(IDS,limits[driver])):
            add(f'{family}_{driver}_{i}',0.,-limit,limit,1 if family=='j' else 2,f'tone{tid}{dctl}To{target}')
    add(f'{family}_fit',0,-2,6,0,fit)
for family,limit,precision in [('chroma',16.,3),('hue',360.,2),('scale',8.,2)]:
    for tone in range(4):
        for handle in range(6):
            add(f'web_{family}_{tone}_{handle}',0.,-limit,limit,precision,f'web{family}{tone}{handle}')
add('global_gain',0.,-3,3,2,'globalChromaGain')
META={c[0]:c for c in CONTROLS}
PARAM_NAMES=tuple(META)
Parameters=make_dataclass('Parameters',[(c[0],int if c[5]==0 else float,c[2]) for c in CONTROLS],frozen=True)
Parameters.__module__=__name__
ADJUST_NAMES=tuple(n for n in PARAM_NAMES if 'fit' not in n and n!='adaptance_half_saturation')

@dataclass(frozen=True)
class Settings:
    lightness: tuple
    chroma: tuple
    adaptance: tuple
    lightness_fit: float
    chroma_fit: float
    global_gain: float
    adaptance_half_saturation: float
    web: WebSettings

def drivers(q, ja, capacity=None):
    q = np.asarray(q, dtype=np.float64)
    c = np.hypot(q[..., 1], q[..., 2])
    if capacity is None:
        from .cube_boundary import default_context
        capacity = default_context().capacity(ja, np.arctan2(q[..., 2], q[..., 1]))
    r = CHROMA_CROSSOVER*np.asarray(capacity, dtype=np.float64)
    scale = np.maximum(c, r)
    a = np.divide(c, scale, out=np.zeros_like(c), where=scale > 0)
    b = np.divide(r, scale, out=np.zeros_like(c), where=scale > 0)
    denominator = a*a+b*b
    visible = (c/np.hypot(c, 2))**2
    high = np.divide(a*a, denominator, out=np.zeros_like(c), where=denominator > 0)
    low = np.divide(b*b, denominator, out=np.zeros_like(c), where=denominator > 0)
    return np.concatenate((np.stack((visible*low, visible*high), axis=-1),
                           q[..., 1:]/np.hypot(c, 2)[..., None]), axis=-1)


class Model(GuardedWarpDisplay):
    def __init__(self, tone, params):
        self.params, self.tone = params, tone
        if not np.isfinite([getattr(params,n) for n in PARAM_NAMES]).all():
            raise ValueError('Colour parameters must be finite')
        self.lightness=tuple(tuple(getattr(params,f'j_{key}_{i}') for key,_,_ in DRIVERS) for i in range(4))
        self.chroma=tuple(tuple(getattr(params,f'c_{key}_{i}') for key,_,_ in DRIVERS) for i in range(4))
        self.adaptance=tuple(tuple(getattr(params,f'adaptance_{axis}_{i}') for axis in 'ab') for i in range(4))
        self.lightness_fit,self.chroma_fit=params.j_fit,params.c_fit
        self.global_gain,self.adaptance_half_saturation=params.global_gain,params.adaptance_half_saturation
        controls={field:tuple(tuple(getattr(params,f'web_{family}_{i}_{j}') for j in range(6)) for i in range(4))
                  for family,field in [('chroma','log_chroma'),('hue','hue_degrees'),('scale','log_scale')]}
        self.settings=Settings(self.lightness,self.chroma,self.adaptance,self.lightness_fit,
                               self.chroma_fit,self.global_gain,self.adaptance_half_saturation,WebSettings(**controls))
        self.web=TonalWeb(self.settings.web,tone)
        self.anchors=100*tone.target[1:]
        self.adaptation=SelectiveAdaptance(self.adaptance,self.anchors,tone.aw,self.adaptance_half_saturation)
        self.active=self.web.active or any(np.any(v) for v in
                                           (self.lightness,self.chroma,self.adaptance,self.global_gain))

    @cached_property
    def boundary(self):
        from .forward import Model
        return Model(self.tone,self).boundary

    def capacity(self,q,ja):
        return self.boundary.capacity(ja,np.arctan2(q[...,2],q[...,1]))

    def terms(self, h):
        p = self.settings
        def evaluate(values, fit):
            return tm.coefficient_weights(h, self.tone.coefficient_knots, self.tone.high, fit) @ np.asarray(values)
        return evaluate(p.lightness, p.lightness_fit), evaluate(p.chroma, p.chroma_fit)

    def warp_state(self, state, tonal_h=None, source_chroma=None, source_energy=None):
        c = np.hypot(state.q[..., 1], state.q[..., 2]) if source_chroma is None else np.asarray(source_chroma)
        h = np.hypot(state.ja, np.sqrt(HK*c)) if tonal_h is None else np.asarray(tonal_h)
        source_energy = state.energy if source_energy is None else source_energy
        transform = (lambda q,e:self.adaptation.apply(q,e,h,c,source_energy)) if self.adaptation.active else None
        q, e = self.web.apply(state.q, state.energy, h, transform=transform)
        return State(q, e)

    def apply_state(self, appearance):
        q = appearance.q
        c = np.hypot(q[..., 1], q[..., 2])
        h = np.hypot(appearance.ja, np.sqrt(HK*c))
        p = self.settings
        energy = appearance.energy if hasattr(appearance, 'energy') else appearance.ja*np.abs(appearance.ja)
        source_energy = energy
        if not self.active:
            return State(q.copy(), np.array(energy, copy=True))
        if any(np.any(v) for v in (p.lightness, p.chroma, p.global_gain)):
            light, chroma = self.terms(h)
            capacity = self.boundary.capacity(appearance.ja, np.arctan2(q[..., 2], q[..., 1]))
            d = drivers(q, appearance.ja, capacity)
            gain = np.exp2(p.global_gain+np.sum(chroma*d, axis=-1))
            out = np.concatenate(((q[..., 0]+np.sum(light*d, axis=-1))[..., None],
                                  q[..., 1:]*gain[..., None]), axis=-1)
            out = np.where((h == 0)[..., None], q, out)
            energy = np.where(np.all(out == q, axis=-1), energy, requested_energy(out, q, appearance.ja))
        else:
            out = q.copy()
        return self.warp_state(State(out, energy), h, c, source_energy)

    def scaled_apply_state(self, q, ja, carried_energy=None):
        q, ja = list(map(S, q)), S(ja)
        p = self.settings
        energy = S(carried_energy) if carried_energy is not None else ja*abs(ja)
        source_energy = energy
        c = norm(q[1:]); h = norm([ja, (HK*c)**.5])
        if not self.active or not h:
            return q, energy
        result = list(q)
        if any(np.any(v) for v in (p.lightness, p.chroma, p.global_gain)):
            angle = np.arctan2(float(q[2]/c), float(q[1]/c)) if c else 0.
            from .cube_boundary import white_response
            if S() < ja < self.boundary.white:
                t = (ja/self.boundary.white)**(1/GAMMA)
                capacity = white_response(self.boundary.fl)*t*(1-t)*float(self.boundary.unit(float(t), angle))
            else:
                capacity = S()
            r = CHROMA_CROSSOVER*capacity
            denominator = c*c+r*r; visibility = c*c/(c*c+4)
            d = [visibility*r*r/denominator if denominator else S(),
                 visibility*c*c/denominator if denominator else S(), *(v/norm([c, 2]) for v in q[1:])]
            def evaluate(values, fit):
                weights = tm.scaled_weights(h, self.tone.coefficient_knots, self.tone.high, fit)
                return [dot(weights, np.asarray(values)[:, i]) for i in range(4)]
            light, chroma = evaluate(p.lightness, p.lightness_fit), evaluate(p.chroma, p.chroma_fit)
            gain = (S(p.global_gain)+dot(chroma, d)).exp2()
            result = [q[0]+dot(light, d), *(v*gain for v in q[1:])]
            if result != q:
                energy = requested_energy(result, q, ja)
        transform = (lambda q,e:self.adaptation.scaled_apply(q,e,h,c,source_energy)) if self.adaptation.active else None
        return self.web.scaled_apply(result, energy, h, transform=transform)

    def scaled_apply(self, q, ja):
        return self.scaled_apply_state(q, ja)[0]

    def warp_targets(self, tonal_row):
        geometry = self.web.geometry; h = geometry.anchors[tonal_row]
        q = np.concatenate(([[h, 0., 0.]], geometry.q[tonal_row]), axis=0)
        energy = np.r_[h*h, geometry.energy[tonal_row]]
        return self.warp_state(State(q, energy), h)

    def target_settings(self, tonal_row, handle, target_ab):
        """Absolute combined page target; -1 denotes the neutral center."""
        if not 0 <= tonal_row < 4 or not -1 <= handle < 6:
            raise ValueError('Warp target index is outside its four-by-seven grid')
        h = self.web.geometry.anchors[tonal_row]
        target = np.asarray(target_ab, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Warp target must be a finite a/b pair')
        if handle == -1:
            adapted = np.array(self.settings.adaptance, float)
            adapted[tonal_row] = target
            SelectiveAdaptance(adapted,self.anchors,self.tone.aw,self.settings.adaptance_half_saturation)
            return replace(self.settings, adaptance=tuple(map(tuple, adapted)))
        return replace(self.settings, web=self.adaptation.target(self.web,tonal_row,handle,target))

    def target_values(self, tonal_row, handle, target_ab):
        settings=self.target_settings(tonal_row,handle,target_ab)
        return self._target_values(settings,tonal_row,handle)

    def saturation_targets(self, tonal_row):
        state=self.warp_targets(tonal_row)
        return project(state.q,state.energy)

    def bounded_saturation_targets(self, tonal_row):
        state=self.warp_targets(tonal_row)
        return bounded_project(state.q,state.energy)

    def target_bounded_saturation_settings(self, tonal_row, handle, target):
        """Convert the bounded display destination to the branch-local solver."""
        return self.target_saturation_settings(tonal_row,handle,bounded_unproject(target))

    def target_bounded_saturation_values(self, tonal_row, handle, target):
        settings=self.target_bounded_saturation_settings(tonal_row,handle,target)
        return self._target_values(settings,tonal_row,handle)

    def target_guarded_saturation_values(self, tonal_row, handle, target):
        return self._target_values(self.target_guarded_saturation_settings(tonal_row,handle,target),tonal_row,handle)

    def target_guarded_hue_values(self, tonal_row, handle, log_chroma, angle):
        return self._target_values(self.target_guarded_hue_settings(tonal_row,handle,log_chroma,angle),tonal_row,handle)

    def guarded_boundary_values(self, tonal_row, handle, angle=None):
        return self._target_values(self.guarded_boundary_settings(tonal_row,handle,angle),tonal_row,handle)

    def target_saturation_settings(self, tonal_row, handle, target):
        """Set a linear saturation destination while preserving the local branch."""
        if not 0 <= tonal_row < 4 or not -1 <= handle < 6:
            raise ValueError('Warp target index is outside its four-by-seven grid')
        target=np.asarray(target,float)
        if target.shape!=(2,) or not np.isfinite(target).all():
            raise ValueError('Warp saturation target must be a finite pair')
        if handle==-1:
            return self.target_settings(tonal_row,handle,self.anchors[tonal_row]*target)
        return replace(self.settings,web=self.adaptation.saturation_target(self.web,tonal_row,handle,target))

    def target_saturation_values(self, tonal_row, handle, target):
        return self._target_values(self.target_saturation_settings(tonal_row,handle,target),tonal_row,handle)

    @staticmethod
    def _target_values(settings, tonal_row, handle):
        if handle==-1:
            return {f'adaptance_{axis}_{tonal_row}':settings.adaptance[tonal_row][i] for i,axis in enumerate('ab')}
        return {f'web_{family}_{tonal_row}_{handle}':getattr(settings.web,field)[tonal_row][handle]
                for family,field in [('chroma','log_chroma'),('hue','hue_degrees'),('scale','log_scale')]}

    def apply_appearance(self, appearance):
        return self.apply_state(appearance).q

    def weights(self,M,fit=4,zero_black=False):
        return tm.coefficient_weights(M,self.tone.coefficient_knots,self.tone.high,fit,zero_black)

    def heights(self,stem):
        return np.array([getattr(self.params,f'{stem}_{i}') for i in range(4)])

    def curve(self,M,stem):
        if stem.startswith('adaptance_'):
            return cardinal_values(M,self.anchors,self.heights(stem))
        family=stem.split('_')[0]
        return self.weights(M,getattr(self.params,family+'_fit'))@self.heights(stem)

    def drivers(self,q,JA=None):
        if JA is None:
            energy=hk_energy(q[...,0],np.hypot(q[...,1],q[...,2]))
            JA=np.sign(energy)*np.sqrt(np.abs(energy))
        return drivers(q,JA,self.capacity(np.asarray(q),JA))

    def stage3(self,q,JA):
        return self.apply_appearance(Appearance(q,JA))

    def stage2_with_ja(self,q,JA):
        value=self.tone.apply_appearance(Appearance(q,JA))
        return value.q,value.ja

    def apply(self,rgb):
        from .forward import Model
        return Model(self.tone,self).apply(rgb)

    def decode(self,q,reference=None):
        # Signed mathematical decoding for chart calibration, not grade output.
        q=np.asarray(q,dtype=float);result=[]
        if q.shape[-1:]!=(3,) or not np.isfinite(q).all():
            raise ValueError('Decode requires finite appearance triples')
        if np.max(np.abs(q),initial=0)<2.**100:
            energy=hk_energy(q[...,0],np.hypot(q[...,1],q[...,2]))
            angle=np.arctan2(q[...,2],q[...,1])
            components=np.concatenate((achromatic_response(energy,self.tone.aw)[...,None],
                q[...,1:]*(self.tone.aw/(1204*eccentricity(angle)))[...,None]),axis=-1)
            values=components@INVERSE.T
            margins=400-np.abs(values)
            if np.any(margins<=0):
                raise ValueError('Mathematical decode requires response magnitudes below 400')
            with np.errstate(divide='ignore'):
                out=np.sign(values)*np.exp((np.log(27.13)+np.log(np.abs(values))-np.log(margins))/.42)/self.tone.fl
            return out,np.isfinite(out).all(axis=-1)
        for row in q.reshape(-1,3):
            energy=hk_energy(S(row[0]),norm(row[1:]))
            c=norm(row[1:]);angle=np.arctan2(float(S(row[2])/c),float(S(row[1])/c)) if c else 0.
            values=[achromatic_response(energy,self.tone.aw),*(S(v)*self.tone.aw/(1204*eccentricity(angle)) for v in row[1:])]
            records=[Response(dot(axis,values),400-abs(dot(axis,values))) for axis in INVERSE]
            if any(r.margin<=0 for r in records):
                raise ValueError('Mathematical decode requires response magnitudes below 400')
            result.append([float(uncompress(r,self.tone.fl)) for r in records])
        out=np.asarray(result).reshape(q.shape)
        return out,np.isfinite(out).all(axis=-1)


def chroma_crossover(capacity):
    """The selector boundary is a fraction of each hue's physical cube ray."""
    return CHROMA_CROSSOVER*np.asarray(capacity)


@lru_cache(maxsize=64)
def model(tone_params,params,space='Linear Rec.709'):
    return Model(tm.model(tone_params,space),params)
