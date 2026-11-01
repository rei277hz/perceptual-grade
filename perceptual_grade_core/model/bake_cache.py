"""Bounded shared row caches for fixed LUT entry, tone and tonal weights.

Only completed rows become visible. Cancellation may leave reusable fixed
work behind, but never publishes a partially calculated LUT. The byte budget
includes registered CPU/GPU lookups and lattice arrays, not output assets.
"""
from collections import OrderedDict
from threading import RLock
import numpy as np

BUDGET=32*1024*1024


class Store:
    def __init__(self,budget=BUDGET):
        self.budget=budget;self.rows=OrderedDict();self.fixed={};self.lock=RLock()

    @property
    def bytes(self):
        return sum(self.fixed.values())+sum(a.nbytes+b.nbytes for a,b,_ in self.rows.values())

    def reserve(self,key,size):
        # Requests repeat these immutable reservations on Blender's main
        # thread. Do not wait for an in-flight worker row calculation.
        if self.fixed.get(key)==size:return
        with self.lock:
            if size<0:raise ValueError('Cache reservation must be nonnegative')
            if sum(self.fixed.values())-self.fixed.get(key,0)+size>self.budget:
                raise MemoryError('Fixed LUT math caches exceed 32 MiB')
            self.fixed[key]=size
            self._evict(0)

    def _evict(self,required):
        while self.rows and self.bytes+required>self.budget:self.rows.popitem(last=False)
        if self.bytes+required>self.budget:raise MemoryError('Fixed LUT math caches exceed 32 MiB')

    def fetch(self,key,indices,total,calculate):
        indices=np.asarray(indices,dtype=np.int64)
        with self.lock:
            if key not in self.rows:
                self._evict(total*17)
                self.rows[key]=(np.empty((total,4),dtype=np.float32),np.zeros(total,dtype=bool),object())
            self.rows.move_to_end(key)
            values,valid,token=self.rows[key];missing=~valid[indices]
            output=values[indices]
            # Keep only the requested-row copy and entry identity during work,
            # so eviction can release the full lattice arrays immediately.
            del values,valid
        if np.any(missing):
            result=np.asarray(calculate(missing),dtype=np.float32)
            if result.shape!=(int(np.count_nonzero(missing)),4) or not np.isfinite(result).all():
                raise ArithmeticError('Invalid cached LUT coordinates')
            output[missing]=result
            with self.lock:
                entry=self.rows.get(key)
                if entry is not None and entry[2] is token:
                    rows=indices[missing]
                    entry[0][rows]=result;entry[1][rows]=True
                del entry
        # Overlapping misses may compute twice; neither job waits for the other.
        # A stale job still owns its result, but cannot resurrect an old entry.
        return output

    def clear(self):
        with self.lock:self.rows.clear();self.fixed.clear()


SHARED=Store()


def tone_key(tone):
    return (bool(tone.active),float(tone.black),float(tone.span),float(tone.slope),float(tone.high),
            float(tone.width),*map(float,tone.knots),*map(float,tone.residual),*map(float,tone.coefficient_knots))


class CachedKernel:
    def __init__(self,kernel,total,store=SHARED):
        self.kernel=kernel;self.total=total;self.store=store;self.tone=tone_key(kernel.tone)
        self.prefix=(kernel.resource_key,total)

    def cached(self,key,indices,calculate):
        return self.store.fetch((self.prefix,*key),indices,self.total,calculate)

    def evaluate(self,rgb,indices,checkpoint=lambda:None):
        k=self.kernel;checkpoint()
        if k.identity:return np.asarray(rgb,dtype=np.float32).copy()
        state=self.cached(('entry','CPU'),indices,lambda missing:k.enter(rgb[missing]))
        checkpoint()
        if k.tone.active:
            state=self.cached(('tone',self.tone),indices,lambda missing:k.tone_state(state[missing]))
        checkpoint();tonal={};p=k.color
        for names,fit in ((('lightness',),p.lightness_fit),(('chroma',),p.chroma_fit)):
            if fit in tonal or not any(np.any(getattr(p,n)) for n in names):continue
            tonal[fit]=self.cached(('weights',self.tone,float(fit)),indices,lambda missing:k.tonal(state[missing],fit))
        checkpoint();graded=k.grade(state,tonal);checkpoint()
        return k.export(graded,state)
