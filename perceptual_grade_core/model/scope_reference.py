"""Numerical comparison scopes aligned to the primary sample grid."""
import numpy as np
from ...color_core.modcam16 import encode_cones
from .saturation import project, bounded_project
from . import scope_analysis as scopes, scope_color as cs, scope_grid


def evaluate(rgba,chain,space,base,to_grade,cloud_points=True):
    """Worker-only comparison math, using the primary snapshot's UV positions."""
    rgb=scopes.apply_chain(rgba[...,:3],chain,space)@to_grade.T
    model=base['model']
    appearance=encode_cones(model.cones(rgb),model.reference)
    q=appearance.q;energy=appearance.ja*np.abs(appearance.ja)
    alpha=np.where(np.isfinite(rgb).all(axis=-1),np.clip(rgba[...,3],0,1),0.)
    mass,x=scope_grid.values(base.get('sample_grid'),alpha.shape)
    return dict(rgb=rgb,appearance=q,energy=energy,saturation=project(q,energy),
                bounded_saturation=bounded_project(q,energy) if cloud_points else None,alpha=alpha,
                sample_mass=mass,sample_x=x,
                distribution=scopes.distribution(q[...,0]/100,alpha,base['model'].plot_max,base.get('sample_grid')),
                vectorscope=cs.vectorscope(rgb,alpha*mass,base['model'].space))


def select(signal,selection,maximum):
    primary_weights=selection['weights']
    weights=primary_weights*signal['alpha']
    opacity=selection.get('opacity_weights',primary_weights)*signal['alpha']
    return dict(signal=signal,primary_weights=primary_weights,weights=weights,
        opacity_weights=opacity,sample_mass=signal.get('sample_mass'),sample_x=signal.get('sample_x'),
        waveform=cs.output_waveform(signal['appearance'],weights,maximum,x=signal.get('sample_x'))['density'],
        vectorscope=cs.vectorscope(signal['rgb'],weights,selection['model'].tone.space))

