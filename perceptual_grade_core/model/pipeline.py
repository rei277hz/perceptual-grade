"""Inspector state after signed Stage 1, input repair and raw appearance entry."""
from . import forward

class Prepared:
    def __init__(self,base,model):
        engine=forward.Model(model.tone,model)
        self.rgb=base['stage2_input_rgb']
        self.entry_rgb=base.get('stage1_rgb')
        self.cones=model.tone.cones(self.rgb)
        self.source=engine.encode_cones(self.cones)
        self.stage2=model.tone.apply_appearance(self.source)

    def apply(self,model):
        engine=forward.Model(model.tone,model)
        if self.entry_rgb is not None and not (model.tone.active or model.active):
            return engine.apply(self.entry_rgb)
        if engine.vector_controls:
            native=engine._core_arrays(self.cones,self.source,self.stage2)
            if native is not None:return engine.from_native(native)
        return engine.apply(self.rgb,entry=False)

