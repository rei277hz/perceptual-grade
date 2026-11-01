"""Select one cube face using the twelve cached cube-edge traces.

An edge stores its hue and reciprocal chroma radius versus neutral response
height. Its two face IDs have an oriented meaning: before/after the edge in
increasing hue. Vertex ties discard zero-width faces, not physical creases.
"""
import numpy as np
from .boundary_table import stencil


def topology(vertices, edges, faces, limits):
    """Candidate rings for seven open height bands and their eight vertex ties.

    Each row holds (left edge, first right edge, second right edge, face).
    At a vertex, a surviving face may have two coincident right edges; retain
    both and the original edge-ID tie order. Padding duplicates the first edge.
    """
    heights=np.sort(vertices)
    samples=np.sort(np.r_[heights,(heights[:-1]+heights[1:])/2])
    rings=np.empty((15,5,4),dtype=np.int32)
    low,high=vertices[limits].T
    for i,t in enumerate(samples):
        active=(t>=vertices[edges[:,0]])&(t<=vertices[edges[:,1]])
        exists=((t>low)|(low==0))&((t<high)|(high==1))
        candidates=[]
        for edge in np.flatnonzero(active&exists[faces[:,1]]):
            face=faces[edge,1];right=np.flatnonzero(active&(faces[:,0]==face))
            if not 1<=len(right)<=2:raise ValueError('Invalid crease adjacency')
            candidates.append((edge,right[0],right[-1],face))
        if not 1<=len(candidates)<=5:raise ValueError('Invalid height-band topology')
        rings[i]=candidates+[candidates[0]]*(5-len(candidates))
    return heights,rings


class Selector:
    def __init__(self, data, dtype=np.float64):
        self.knots = np.asarray(data['edge_knots'], dtype=dtype)
        self.coefficients = np.asarray(data['edge_coefficients'], dtype=dtype)
        self.vertices = np.asarray(data['vertex_t'], dtype=dtype)
        self.edges = np.asarray(data['edges'])
        self.faces = np.asarray(data['edge_faces'])
        self.limits = np.asarray(data['face_limits'])
        self.heights,self.topology=topology(self.vertices,self.edges,self.faces,self.limits)

    def region(self,t):
        band=np.clip(np.searchsorted(self.heights,t,side='right')-1,0,7)
        return 2*band+(t!=self.heights[band])

    def curve(self,t,edge,channel):
        low=self.vertices[self.edges[edge,0]];high=self.vertices[self.edges[edge,1]]
        x=np.clip((t-low)/(high-low),0,1)
        i,w=stencil(self.knots,x)
        return np.sum(self.coefficients[np.asarray(edge)[...,None],i,channel]*w,axis=-1)

    def curves(self, t):
        t = np.asarray(t, dtype=self.coefficients.dtype)
        low, high = self.edges.T
        x = ((t[..., None]-self.vertices[low]) /
             (self.vertices[high]-self.vertices[low]))
        i, w = stencil(self.knots, np.clip(x, 0, 1))
        values = np.sum(self.coefficients[np.arange(12)[:, None], i] *
                        w[..., None], axis=-2)
        return values, (x >= 0) & (x <= 1)

    def left(self,t,hue,region):
        candidates=self.topology[region,:,0]
        angles=self.curve(t[:,None],candidates,0)%(2*np.pi)
        distance=(hue[:,None]-angles)%(2*np.pi)
        slot=distance.argmin(-1);row=np.arange(len(t))
        return self.topology[region,slot],distance[row,slot],angles[row,slot]

    def select(self, t, hue):
        """Face, angular fraction, and the two shared edge traces.

        Output arrays have the broadcast input shape. No face radius is
        evaluated or compared in deciding which face owns a hue.
        """
        t, hue = np.broadcast_arrays(np.asarray(t, dtype=self.coefficients.dtype),
                                    np.asarray(hue, dtype=self.coefficients.dtype))
        shape = t.shape
        t, hue = t.ravel(), hue.ravel()
        region=self.region(t)
        chosen,dl,left_angle=self.left(t,hue,region)
        left,r0,r1,face=chosen.T
        right=r0.copy();right_angle=self.curve(t,right,0)%(2*np.pi)
        dr=(right_angle-hue)%(2*np.pi)
        ties=np.flatnonzero(r0!=r1)
        if len(ties):
            angle=self.curve(t[ties],r1[ties],0)%(2*np.pi)
            distance=(angle-hue[ties])%(2*np.pi);take=distance<dr[ties];ids=ties[take]
            right[ids]=r1[ids];right_angle[ids]=angle[take];dr[ids]=distance[take]
        fraction = np.divide(dl, dl+dr, out=np.zeros_like(dl), where=dl+dr > 0)
        values = (face,fraction,left_angle,right_angle,
                  self.curve(t,left,1),self.curve(t,right,1))
        return tuple(value.reshape(shape) for value in values)
