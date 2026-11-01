"""Independent CPU grade in raw coordinates, with validated native AP1 export."""
import numpy as np
from . import linear
from ...color_core import modcam16 as responses
from .export import input_repair
from .cube_boundary import context
from ...color_core.scaled import Scalar as S, dot
from .hk_guard import GradedState, guard as hk_guard, guard_scaled as hk_guard_scaled
CAT16=np.asarray(linear.CAT16)
SCHEMA=14

class Model:
    prepared=False

    def __init__(self, tone, color=None, matrix=linear.I3):
        from . import color as creative
        if not np.array_equal(tone.w, np.asarray(linear.xyz(linear.white('Linear Rec.709')))) or tone.settings.reference_nits != 203.:
            raise ValueError('Only cached D65 / 203 nits is supported')
        self.tone=tone;self.color=color or creative.Model(tone,creative.Parameters())
        self.space=tone.space
        self.white,self.white_cones=tone.w,tone.wc
        self.rgb_to_xyz,self.xyz_to_rgb=tone.rgb_to_xyz,tone.xyz_to_rgb
        self.to_acescg=(np.asarray(linear.xyz_to_rgb('ACEScg'))
                       @np.asarray(linear.bradford(linear.white(self.space),linear.white('ACEScg')))@self.rgb_to_xyz)
        self.acescg_y=np.asarray(linear.rgb_to_xyz('ACEScg'))[1]
        self.linear=np.asarray(matrix,dtype=float)
        if self.linear.shape!=(3,3) or not np.isfinite(self.linear).all():
            raise ValueError('Authoritative XYZ matrix must be finite and 3 by 3')
        self.cones_to_xyz=np.asarray(linear.ICAT16)@np.diag(self.white_cones)
        self.native_matrix=(np.asarray(linear.xyz_to_rgb('ACEScg'))
                            @np.asarray(linear.bradford(linear.white(self.space),linear.white('ACEScg')))
                            @self.cones_to_xyz)
        self.fl,self.aw=tone.fl,tone.aw
        p=self.color.settings
        self.vector_controls=(np.max(np.abs(self.linear))<2.**200 and
                              np.max(np.abs(p.lightness))<2.**100 and np.max(np.abs(p.adaptance))<2.**100 and
                              np.max(np.abs(p.chroma))<=64 and abs(p.global_gain)<=128 and
                              np.max(np.abs(p.web.log_chroma))<=64 and np.max(np.abs(p.web.log_scale))<=64)

    @property
    def boundary(self):return context(self.fl,self.aw,self.native_matrix)

    def encode_cones(self,cones):
        boundary=self.boundary
        return responses.encode_cones(cones,self.tone.reference,(boundary.fl,boundary.aw))

    def to_native(self,rgb,apply_linear=False):
        """Convert and optionally grade raw input before its one AP1 repair."""
        values=np.asarray(rgb,dtype=float)
        if values.shape[-1:]!=(3,) or not np.isfinite(values).all():
            raise ValueError('Input must contain finite triples')
        matrix=self.linear if apply_linear else np.eye(3)
        return self._input_native(values,self.rgb_to_xyz,matrix,np.ones(3))

    def _input_native(self,values,before,matrix,neutral):
        native_xyz=self.boundary.matrix@(CAT16/self.white_cones[:,None])
        d=matrix[1,1]
        if (np.max(abs(values),initial=0)<2.**700 and np.all((values==0)|(abs(values)>=2.**-700))
                and np.max(abs(matrix))<2.**200 and np.all((matrix==0)|(abs(matrix)>=2.**-200))):
            delta=((values-values[...,1,None]*neutral)@before.T)@matrix.T
            delta+=values[...,1,None]*((matrix-d*np.eye(3))@self.white)
            return input_repair(values[...,1,None]*d+delta@native_xyz.T,self.acescg_y)
        result=[]
        for row in values.reshape(-1,3):
            delta=[dot(axis,[S(v)-S(row[1])*n for v,n in zip(row,neutral)]) for axis in before]
            delta=[dot(axis,delta)+S(row[1])*dot([S(v)-(S(d) if i==j else S()) for j,v in enumerate(axis)],self.white)
                   for i,axis in enumerate(matrix)]
            native=[S(row[1])*d+dot(axis,delta) for axis in native_xyz]
            amplitude=max(map(abs,native))
            unit=np.array([float(v/amplitude) if amplitude else 0. for v in native])
            repaired=input_repair(unit,self.acescg_y);upper=max(repaired)
            factor=min(amplitude,S(np.nextafter(1024.,0.))/upper) if upper else S()
            result.append([S(v)*factor for v in repaired])
        scaled=any(v and abs(v)<S(2.**-1021) for row in result for v in row)
        return np.asarray(result,dtype=object if scaled else float).reshape(values.shape)

    def _scaled_entry(self,values,export,checkpoint):
        inverse=np.linalg.inv(self.boundary.matrix)
        result=[]
        for row in values.reshape(-1,3):
            if checkpoint is not None:checkpoint()
            cones=[S(row[1])+dot(axis,[S(v)-row[1] for v in row]) for axis in inverse]
            q,ja=responses.coordinates([responses.compress(v,self.boundary.fl) for v in cones],self.boundary.aw)
            if cones[0]==cones[1]==cones[2]:q=[ja,S(),S()]
            result.append([float(v) for v in self.core(cones,export,checkpoint)])
        return np.asarray(result).reshape(values.shape)

    def from_native(self,native):
        return native[...,1,None]+(native-native[...,1,None])@np.linalg.inv(self.to_acescg).T

    def prepare_native(self,rgb,checkpoint=None):
        if checkpoint is not None:checkpoint()
        return input_repair(rgb,self.acescg_y)

    def prepare_input(self,rgb,checkpoint=None):
        return self.from_native(self.prepare_native(self.to_native(rgb),checkpoint))

    @property
    def matrix_native(self):
        if np.array_equal(self.linear,np.eye(3)):return np.eye(3)
        return (self.boundary.matrix@(CAT16/self.white_cones[:,None])@self.linear
                @self.cones_to_xyz@np.linalg.inv(self.boundary.matrix))

    def linear_input(self,rgb):
        return self._input_native(np.asarray(rgb),self.cones_to_xyz@np.linalg.inv(self.boundary.matrix),
                                  self.linear,np.ones(3))

    def enter_native(self,rgb,checkpoint=None):
        """Repair and encode raw appearance, without an intermediate RGB decode."""
        values=np.asarray(rgb);inverse=np.linalg.inv(self.boundary.matrix)
        if values.dtype==object:
            rows=[];energies=[]
            for row in values.reshape(-1,3):
                if checkpoint is not None:checkpoint()
                cones=[S(row[1])+dot(axis,[S(v)-row[1] for v in row]) for axis in inverse]
                q,j=responses.coordinates([responses.compress(v,self.boundary.fl) for v in cones],self.boundary.aw)
                if cones[0]==cones[1]==cones[2]:q=[j,S(),S()]
                e=j*abs(j)
                rows.append(q);energies.append(e)
            return responses.State(np.asarray(rows,dtype=object).reshape(values.shape),
                                   np.asarray(energies,dtype=object).reshape(values.shape[:-1]))
        values=input_repair(values,self.acescg_y)
        cones=values[...,1,None]+(values-values[...,1,None])@inverse.T
        source=self.encode_cones(cones)
        return responses.State(source.q,source.ja*abs(source.ja))

    def grade_state(self,state,checkpoint=None):
        """Stage 2/3 raw handoff with fixed post-tone energy for RGB export."""
        if checkpoint is not None:checkpoint()
        if state.q.dtype!=object and self.vector_controls:
            base=self.tone.apply_appearance(state)
            result=self.color.apply_state(base)
            reference=base.ja*abs(base.ja) if self.tone.active else state.energy.copy()
            return GradedState(result.q,result.energy,reference)
        rows=[];energies=[];references=[]
        for q,j,e in zip(state.q.reshape(-1,3),state.ja.reshape(-1),state.energy.reshape(-1)):
            if checkpoint is not None:checkpoint()
            base,j=self.tone.scaled_apply(list(map(S,q)),S(j))
            reference=j*abs(j) if self.tone.active else S(e)
            request,energy=self.color.scaled_apply_state(base,j,reference)
            rows.append(request);energies.append(energy);references.append(reference)
        return GradedState(np.asarray(rows,dtype=object).reshape(state.q.shape),
                           np.asarray(energies,dtype=object).reshape(state.energy.shape),
                           np.asarray(references,dtype=object).reshape(state.energy.shape))

    def guarded_appearance(self, appearance, checkpoint=None):
        """Apply Stage 3 and export guards without decoding through RGB.

        The returned tuple is ``(q, energy, tested_rgb)``.  ``q`` and
        ``energy`` are the exact state consumed by the cube exporter; the RGB
        value is the boundary guard's tested value and is retained for
        vectorscope diagnostics.
        """
        if not self.vector_controls:
            return None
        if np.any((appearance.ja!=0)&(abs(appearance.ja)<2.**-200)):
            return None
        result = self.color.apply_state(appearance)
        if np.any((result.q[...,0]>0)&(result.q[...,0]<2.**-200)):
            return None
        reference = appearance.ja * abs(appearance.ja)
        request, energy = hk_guard(result.q, result.energy, reference)
        return self.boundary.guard(request, energy, checkpoint)

    def release_state(self,state,checkpoint=None,guard=False):
        """Decode raw handoffs; H-K repair then cube projection at RGB export."""
        reference=getattr(state,'reference_energy',None)
        if state.q.dtype!=object:
            q,e=state.q,state.energy
            if guard:
                if reference is not None:q,e=hk_guard(q,e,reference)
                _,_,rgb=self.boundary.guard(q,e,checkpoint)
                return rgb
            return self.boundary.decode(q,e)
        out=[]
        references=np.asarray(reference).reshape(-1) if reference is not None else [None]*state.energy.size
        for q,e,ref in zip(state.q.reshape(-1,3),state.energy.reshape(-1),references):
            if checkpoint is not None:checkpoint()
            q=list(q)
            if guard:
                if ref is not None:q,e=hk_guard_scaled(q,e,ref)
                _,_,rgb=self.boundary.guard_scaled(q,e)
            else:rgb=self.boundary.decode_scaled(q,e)
            out.append(rgb)
        values=np.asarray(out,dtype=object).reshape(state.q.shape)
        tiny=any(v and abs(v)<S(2.**-1021) for v in values.flat)
        return values if tiny else values.astype(float)

    def core(self,cones,export=True,checkpoint=None):
        boundary=self.boundary
        encoded=[responses.compress(v,boundary.fl) for v in cones]
        q,ja=responses.coordinates(encoded,boundary.aw)
        if cones[0]==cones[1]==cones[2]:q=[ja,S(),S()]
        base,j=self.tone.scaled_apply(q,ja)
        request,energy=self.color.scaled_apply_state(base,j)
        if checkpoint is not None:checkpoint()
        if export:
            request,energy=hk_guard_scaled(request,energy,j*abs(j))
            _,_,rgb=self.boundary.guard_scaled(request,energy)
            return rgb
        return self.boundary.decode_scaled(request,energy)

    def _core_arrays(self,cones,source=None,stage2=None,checkpoint=None,export=True):
        source=self.encode_cones(cones) if source is None else source
        stage2=self.tone.apply_appearance(source) if stage2 is None else stage2
        if np.any((stage2.ja!=0)&(abs(stage2.ja)<2.**-200)):return None
        if export:
            guarded=self.guarded_appearance(stage2,checkpoint)
            if guarded is None:return None
            _,_,rgb=guarded
            return rgb
        result=self.color.apply_state(stage2)
        request,energy=result.q,result.energy
        if np.any((request[...,0]>0)&(request[...,0]<2.**-200)):return None
        return self.boundary.decode(request,energy)

    def apply_native(self,rgb,checkpoint=None,export=True,entry=True,apply_linear=True):
        values=np.asarray(rgb)
        internal_scaled=values.dtype==object and all(isinstance(v,S) for v in values.flat)
        if not internal_scaled:values=np.asarray(values,dtype=float)
        if values.shape[-1:]!=(3,) or (not internal_scaled and not np.isfinite(values).all()):
            raise ValueError('Input must contain finite triples')
        if checkpoint is not None:checkpoint()
        matrix=self.linear if apply_linear else np.eye(3)
        if entry:
            if apply_linear:
                values=self._input_native(values,self.cones_to_xyz@np.linalg.inv(self.boundary.matrix),matrix,np.ones(3))
            if export and not (self.tone.active or self.color.active):
                return np.asarray(values,dtype=float) if values.dtype==object else input_repair(values,self.acescg_y)
            if values.dtype==object:return self._scaled_entry(values,export,checkpoint)
            values=self.prepare_native(values,checkpoint)
            matrix=np.eye(3)
        # Stack handoffs decode raw signed appearance; only forward exports guard.
        to_xyz=self.cones_to_xyz@np.linalg.inv(self.boundary.matrix)
        to_cones=CAT16/self.white_cones[:,None]
        d=matrix[1,1]
        if self.vector_controls:
            change=((values-values[...,1,None])@to_xyz.T)@matrix.T
            change+=values[...,1,None]*((matrix-d*np.eye(3))@self.white)
            cones=values[...,1,None]*d+change@to_cones.T
            if np.all((cones==0)|((abs(cones)>=2.**-200)&(abs(cones)<=2.**300))):
                result=self._core_arrays(cones,checkpoint=checkpoint,export=export)
                if result is not None:return result
        result=[]
        for i,row in enumerate(values.reshape(-1,3)):
            if checkpoint is not None:checkpoint(i/max(1,values.size//3))
            change=[dot(axis,[S(v)-row[1] for v in row]) for axis in to_xyz]
            change=[dot(axis,change)+S(row[1])*dot([S(v)-(S(d) if i==j else S()) for j,v in enumerate(axis)],self.white)
                    for i,axis in enumerate(matrix)]
            cones=[S(row[1])*d+dot(axis,change) for axis in to_cones]
            result.append([float(v) for v in self.core(cones,export,checkpoint)])
        return np.asarray(result,dtype=float).reshape(values.shape)

    def apply(self,rgb,checkpoint=None,export=True,entry=True):
        if entry:native=self.to_native(rgb,apply_linear=True)
        else:
            rgb=np.asarray(rgb,dtype=float)
            native=rgb[...,1,None]+(rgb-rgb[...,1,None])@self.to_acescg.T
        return self.from_native(self.apply_native(native,checkpoint,export,entry,apply_linear=not entry))
