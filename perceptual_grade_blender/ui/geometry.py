"""Canvas geometry and clipping shared by drawing, hit tests and unit tests."""

from ..integration.components import Component

def create(_application):


    from dataclasses import dataclass
    import math

    FINE_GAIN = .1


    def in_rect(point, rect):
        x, y, w, h = rect
        return x <= point[0] <= x+w and y <= point[1] <= y+h

    def diagram_layout(region_width,region_height,scale,requested_size,pos_x,pos_y):
        """Preserve the old xy chart size, moving readouts into a right column."""
        s=scale;size=min((requested_size-68)*s,region_width-234*s,region_height-134*s)
        if size<232*s:return None
        w=size+210*s;h=size+110*s
        x=max(12*s,min(pos_x*s,region_width-w-12*s));y=max(12*s,min(pos_y*s,region_height-h-12*s))
        chart=Chart(x+48*s,y+30*s,size)
        pad=(chart.left+size-93*s,chart.bottom+size-93*s,88*s,88*s)
        return dict(chart=chart,rect=(x,y,w,h),scale=s,
            linear_tab=(x+135*s,y+h-27*s,49*s,22*s),tone_tab=(x+186*s,y+h-27*s,49*s,22*s),color_tab=(x+237*s,y+h-27*s,49*s,22*s),
            header=(x,y+h-32*s,w,32*s),close=(x+w-28*s,y+h-27*s,22*s,22*s),
            exposure=(chart.left+91*s,chart.bottom+size+8*s,size-91*s,28*s),
            info=(chart.left+size+18*s,chart.bottom,132*s,size),
            vectorscope=(chart.left+size+18*s,chart.bottom+size-132*s,132*s,132*s),
            balance=pad,balance_axes=(pad[0]+16*s,pad[1]+17*s,56*s,56*s))


    def tone_layout(region_width,region_height,scale,requested_size,pos_x,pos_y,reference_connected=False):
        """One row: input waveform, target waveform, rectangular tone curve."""
        Plot = _application.component('ui.color_geometry').Plot
        s=scale;footer=12*s if reference_connected else 0
        w=min(requested_size*s,region_width-24*s);h=min(338*s+footer,region_height-24*s)
        if w<360*s or h<300*s+footer:return None
        x=max(12*s,min(pos_x*s,region_width-w-12*s));y=max(12*s,min(pos_y*s,region_height-h-12*s))
        top=y+h-129*s;bottom=y+85*s+footer;ph=top-bottom
        wave_w=(w-96*s)*.28;curve_w=w-96*s-2*wave_w
        before=(x+24*s,bottom,wave_w,ph)
        after=(before[0]+wave_w+24*s,bottom,wave_w,ph)
        curve=(after[0]+wave_w+24*s,y+43*s+footer,curve_w+7*s,top-y-43*s-footer)
        return dict(rect=(x,y,w,h),scale=s,header=(x,y+h-32*s,w,32*s),
            close=(x+w-28*s,y+h-27*s,22*s,22*s),
            linear_tab=(x+135*s,y+h-27*s,49*s,22*s),tone_tab=(x+186*s,y+h-27*s,49*s,22*s),
            color_tab=(x+237*s,y+h-27*s,49*s,22*s),
            exposure=(x+17*s,y+h-73*s,w*.45,28*s),
            peak=(x+w*.57,y+h-72*s,w*.39,27*s),
            before=before,after=after,chart=Plot(curve,0,1,0,1),
            fit=(before[0],y+42*s+footer,2*wave_w+24*s,30*s),
            fit_track=(before[0],y+47*s+footer,2*wave_w+24*s,12*s))


    def exposure_point(value, box):
        x, y, w, _ = box['exposure']
        return x+(value+6)/12*w, y+3*box['scale']


    def balance_point(cr, yb, box):
        x, y, w, h = box['balance_axes']
        return x+(-cr+2)/4*w, y+(yb+2)/4*h


    def gain_drag(mode, params, dx, dy, box):
        """Relative fine dragging: no click-to-jump, modifiers never alter gain."""
        clamp = lambda v, lo, hi: max(lo, min(hi, v))
        if mode == 'exposure':
            return dict(exposure=clamp(params.exposure+dx*12/box['exposure'][2]*FINE_GAIN, -6, 6))
        if mode == 'balance':
            _, _, w, h = box['balance_axes']
            return dict(cr_balance=clamp(params.cr_balance-dx*4/w*FINE_GAIN, -2, 2),
                        yb_balance=clamp(params.yb_balance+dy*4/h*FINE_GAIN, -2, 2))
        raise ValueError('Unknown gain control: {control}'.format(control=mode))


    @dataclass(frozen=True)
    class Chart:
        left: float
        bottom: float
        size: float
        low: float = 0.
        high: float = .9

        def screen(self, p):
            k = self.size/(self.high-self.low)
            return self.left+(p[0]-self.low)*k, self.bottom+(p[1]-self.low)*k

        def xy(self, p):
            k = (self.high-self.low)/self.size
            return self.low+(p[0]-self.left)*k, self.low+(p[1]-self.bottom)*k

        def contains(self, p, margin=0):
            return self.left-margin <= p[0] <= self.left+self.size+margin and self.bottom-margin <= p[1] <= self.bottom+self.size+margin

        def clip(self, a, b):
            """Liang-Barsky, in xy coordinates; rejects undefined/infinite points."""
            if a is None or b is None or not all(math.isfinite(v) for v in (*a, *b)):
                return None
            dx, dy = b[0]-a[0], b[1]-a[1]
            lo, hi = 0., 1.
            for p, q in ((-dx, a[0]-self.low), (dx, self.high-a[0]),
                         (-dy, a[1]-self.low), (dy, self.high-a[1])):
                if abs(p) < 1e-15:
                    if q < 0:
                        return None
                elif p < 0:
                    lo = max(lo, q/p)
                else:
                    hi = min(hi, q/p)
            if lo > hi:
                return None
            return self.screen((a[0]+lo*dx, a[1]+lo*dy)), self.screen((a[0]+hi*dx, a[1]+hi*dy))


    def hit_handle(point, chart, geometry, radius=12):
        candidates = [('white', 1, geometry['source'])]
        for axis in ('cr', 'yb'):
            candidates.extend((axis, s, p) for s, p in zip((1, -1), geometry['endpoints'][axis]))
        eligible = []
        for index, (kind, sign, p) in enumerate(candidates):
            q = chart.screen(p)
            d = math.hypot(q[0]-point[0], q[1]-point[1])
            if d <= radius and chart.contains(q, radius):
                eligible.append((d, index, kind, sign))
        if not eligible:
            return None
        _, _, kind, sign = min(eligible)
        return kind, sign

    def _set_export_FINE_GAIN(value):
        nonlocal FINE_GAIN
        FINE_GAIN = value

    def _set_export_in_rect(value):
        nonlocal in_rect
        in_rect = value

    def _set_export_diagram_layout(value):
        nonlocal diagram_layout
        diagram_layout = value

    def _set_export_tone_layout(value):
        nonlocal tone_layout
        tone_layout = value

    def _set_export_exposure_point(value):
        nonlocal exposure_point
        exposure_point = value

    def _set_export_balance_point(value):
        nonlocal balance_point
        balance_point = value

    def _set_export_gain_drag(value):
        nonlocal gain_drag
        gain_drag = value

    def _set_export_Chart(value):
        nonlocal Chart
        Chart = value

    def _set_export_hit_handle(value):
        nonlocal hit_handle
        hit_handle = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Chart': lambda: Chart,
        'FINE_GAIN': lambda: FINE_GAIN,
        'balance_point': lambda: balance_point,
        'dataclass': lambda: dataclass,
        'diagram_layout': lambda: diagram_layout,
        'exposure_point': lambda: exposure_point,
        'gain_drag': lambda: gain_drag,
        'hit_handle': lambda: hit_handle,
        'in_rect': lambda: in_rect,
        'math': lambda: math,
        'tone_layout': lambda: tone_layout,
    }, writers={
        'FINE_GAIN': _set_export_FINE_GAIN,
    'in_rect': _set_export_in_rect, 'diagram_layout': _set_export_diagram_layout, 'tone_layout': _set_export_tone_layout, 'exposure_point': _set_export_exposure_point, 'balance_point': _set_export_balance_point, 'gain_drag': _set_export_gain_drag, 'Chart': _set_export_Chart, 'hit_handle': _set_export_hit_handle})
