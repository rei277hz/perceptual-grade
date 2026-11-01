"""Analytic nested 33/65-cube baking and matching tetrahedral evaluation, without bpy.

Stack cores include internal matrices; native nodes own endpoint matrices. Cubes use
fixed native ACEScg/ACES-white coordinates on a nonnegative log1p grid. The
analytic grading coordinates are unchanged. Input is projected into the cube
before lookup; export is guarded during baking, with no output repair.
"""


from functools import lru_cache
import hashlib
import math
import numpy as np
from . import linear
from .export import input_repair

SCHEMA=30
SIZE=65
PREVIEW_SIZE=33
PREVIEW_STRIDE=(SIZE-1)//(PREVIEW_SIZE-1)
CELLS=SIZE-1
SPACE='ACEScg'
CANONICAL_SPACE='Linear Rec.709'
MINIMUM=0.
MAXIMUM=1024.
KAPPA=.01
LOG_RANGE=math.log1p(MAXIMUM/KAPPA)
ATLAS_WIDTH=1024
ATLAS_HEIGHT=(SIZE**3+ATLAS_WIDTH-1)//ATLAS_WIDTH
BLOCK=4096


def atlas_height(size,planes=1):
    return (planes*size**3+ATLAS_WIDTH-1)//ATLAS_WIDTH


@lru_cache(maxsize=8)
def basis(space=CANONICAL_SPACE):
    """Model RGB ↔ fixed LUT RGB, including white adaptation when required.

    GradeSpace uses canonical Rec.709 coordinates even when its CAM white is
    different. Named analytic RGB spaces are also supported by numerical
    tools; their chromatic adaptation is part of the coordinate conversion.
    """
    source=CANONICAL_SPACE if isinstance(space,linear.GradeSpace) else space
    to_lut=(np.asarray(linear.xyz_to_rgb(SPACE))
            @np.asarray(linear.bradford(linear.white(source),linear.white(SPACE)))
            @np.asarray(linear.rgb_to_xyz(source)))
    from_lut=np.linalg.inv(to_lut)
    to_lut.flags.writeable=False;from_lut.flags.writeable=False
    return to_lut,from_lut


TO_LUT,FROM_LUT=basis()
AP1_Y=np.asarray(linear.rgb_to_xyz('ACEScg'))[1]


def model_space(model):
    if hasattr(model,'space'):return model.space
    if hasattr(model,'tone'):return model.tone.space
    return CANONICAL_SPACE


def pack_rgbe(rgb):
    """Signed RGB mantissas plus an exact shared base-2 exponent.

    This is lossless for normal float32 RGB at comparable channel magnitudes.
    Half-float image transport then has relative, rather than scene-dependent
    absolute, precision. Linear RGB is recovered *before* interpolation.
    """
    rgb=np.asarray(rgb,dtype=np.float32)
    magnitude=np.maximum(np.maximum(abs(rgb[...,0]),abs(rgb[...,1])),abs(rgb[...,2]))
    _,exponent=np.frexp(magnitude)
    result=np.empty((*rgb.shape[:-1],4),dtype=np.float32)
    np.ldexp(rgb,-exponent[...,None],out=result[...,:3]);result[...,3]=exponent
    return result


def unpack_rgbe(rgba):
    rgba=np.asarray(rgba,dtype=np.float32)
    return np.ldexp(rgba[...,:3],rgba[...,3].astype(np.int32)[...,None])


def encode(rgb):
    """Map nonnegative linear RGB to addresses, after address-only bounding."""
    x=np.asarray(rgb,dtype=float)
    if np.any(x<0):raise ValueError('LUT address shaper requires nonnegative coordinates')
    return np.log1p(x/KAPPA)/LOG_RANGE


def decode(value):
    """Inverse input shaper. Atlas output values already are linear RGB."""
    return KAPPA*np.expm1(np.asarray(value,dtype=float)*LOG_RANGE)


def signature(model,size=SIZE):
    key=(SCHEMA,size,SPACE,MINIMUM,MAXIMUM,KAPPA,model_space(model),
         'log1p-linear-rgb-tetrahedral',model.bake_key)
    return hashlib.sha256(repr(key).encode('utf8')).hexdigest()


@lru_cache(maxsize=2)
def axis(size=SIZE):
    if size==PREVIEW_SIZE:return axis()[::PREVIEW_STRIDE]
    if size!=SIZE:raise ValueError('Unsupported LUT grid size')
    values=decode(np.linspace(0,1,SIZE))
    values[0]=MINIMUM;values[-1]=MAXIMUM
    values.flags.writeable=False
    return values


@lru_cache(maxsize=2)
def lattice(size=SIZE):
    grid=np.stack(np.meshgrid(axis(size),axis(size),axis(size),indexing='ij'),-1).reshape(-1,3)
    grid.flags.writeable=False
    return grid


@lru_cache(maxsize=1)
def refinement_indices():
    """Disjoint preview/refinement rows in the final B-fast lattice."""
    stride=PREVIEW_STRIDE
    shared=np.arange(SIZE**3).reshape(SIZE,SIZE,SIZE)[::stride,::stride,::stride].ravel()
    mask=np.ones(SIZE**3,dtype=bool);mask[shared]=False
    remaining=np.flatnonzero(mask)
    shared.flags.writeable=False;remaining.flags.writeable=False
    return shared,remaining


def cell_coordinates(rgb,size=SIZE):
    """Cell address and barycentric fractions in physical linear RGB.

    Repair native input before addressing; every fraction is in [0,1].
    """
    rgb=input_repair(rgb,AP1_Y)
    cells=size-1
    position=cells*encode(np.clip(rgb,MINIMUM,MAXIMUM))
    i=np.clip(np.floor(position),0,cells-1).astype(np.int64)
    values=axis(size);lo=values[i];hi=values[i+1]
    return i,np.clip((rgb-lo)/(hi-lo),0,1)


def tetrahedral(table,rgb):
    """Six tetrahedra with the common 000--111 diagonal, four vertex reads.

    The nonuniform grid locates the cell. Its actual linear RGB endpoints
    define the fractions, retaining exact affine interpolation in real
    arithmetic. Ties prefer R, then G, then B.
    """
    original=np.asarray(rgb,dtype=float);shape=original.shape
    values=np.asarray(table)
    size=round((values.size/3)**(1/3))
    values=values.reshape(size,size,size,3)
    x=original.reshape(-1,3);i,f=cell_coordinates(x,size)
    order=np.argsort(-f,axis=-1,kind='stable')
    first=np.eye(3,dtype=np.int64)[order[:,0]]
    second=np.eye(3,dtype=np.int64)[order[:,1]]
    fs=np.take_along_axis(f,order,axis=-1)
    corners=(i,i+first,i+first+second,i+1)
    vertices=[values[c[:,0],c[:,1],c[:,2]] for c in corners]
    # Equivalent convex interpolation with differences; preserve small edges.
    out=vertices[0].astype(float,copy=True)
    for a,b,fraction in zip(vertices,vertices[1:],fs.T):
        out+=(b.astype(float)-a.astype(float))*fraction[:,None]
    return out.reshape(shape)


class Cube:
    """Native ACEScg samples with apply() in the model's RGB coordinates."""
    def __init__(self,values,key,space=CANONICAL_SPACE,size=SIZE):
        axis(size)  # Validate the supported nested grids.
        self.size=size
        self.values=np.asarray(values,dtype=np.float32).reshape(size,size,size,3)
        self.signature=key
        self.diagnostics={}
        self.space=space
        self.to_lut,self.from_lut=basis(space)

    def apply(self,rgb):
        return tetrahedral(self.values,np.asarray(rgb)@self.to_lut.T)@self.from_lut.T

    def pixels(self):
        result=np.ones((atlas_height(self.size),ATLAS_WIDTH,4),np.float32)
        result.reshape(-1,4)[:self.size**3,:3]=self.values.reshape(-1,3)
        return result

    def packed_pixels(self):
        """Compositor base and three edge planes, each with its own RGBE.

        Half-float vertex subtraction would erase small first-cell slopes.
        Pack already-computed edge differences instead; runtime still reads
        only one base and three selected edges from this single image.
        """
        fields=[self.values]
        for axis in range(3):
            edge=np.zeros_like(self.values)
            lo=[slice(None)]*3;hi=lo.copy()
            lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
            np.subtract(self.values[tuple(hi)],self.values[tuple(lo)],out=edge[tuple(lo)])
            fields.append(edge)
        result=np.zeros((atlas_height(self.size,4),ATLAS_WIDTH,4),np.float32)
        count=self.size**3
        for plane,field in enumerate(fields):
            result.reshape(-1,4)[plane*count:(plane+1)*count]=pack_rgbe(field.reshape(-1,3))
        return result
