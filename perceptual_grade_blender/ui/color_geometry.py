"""Pure geometry and hit arbitration for the compact contribution inspector."""

from ..integration.components import Component

def create(_application):


    from dataclasses import dataclass
    import math
    import numpy as np
    from ...perceptual_grade_core.model import color as ch
    controls = _application.component('model.controls')
    in_rect = _application.component('ui.geometry').in_rect; FINE_GAIN = _application.component('ui.geometry').FINE_GAIN


    def slider_fields(family):
        if family=='WARP':return [('global1','adaptance_half_saturation')]
        fields=[('fit',{'J':'j_fit','C':'c_fit'}[family])]
        if family=='C':fields.append(('global1','global_gain'))
        return fields


    def slider_rows(field,width,s):
        meta=controls.meta(field);precision=meta[5]
        value_chars=max(len(f'{v:.{precision}f}') for v in meta[3:5])
        budget=max(9,int((width-(value_chars*5.5+8)*s)/(4.5*s)))
        wrap = _application.component('localization.i18n').wrap
        return wrap(meta[1],budget)


    def layout(rw,rh,s,requested,x,y,family='J',reference_connected=False):
        footer=12*s if reference_connected else 0
        h=min((528 if family=='WARP' else 438)*s+footer,rh-24*s)
        # Warp has one full-height square; all controls are inside it.
        desired_width=h-119*s-footer+38*s if family=='WARP' else requested*s
        w=min(max(360*s,desired_width) if family=='WARP' else desired_width,rw-24*s)
        if w<360*s or h<414*s+footer:return None
        x=max(12*s,min(x*s,rw-w-12*s));y=max(12*s,min(y*s,rh-h-12*s))
        left=x+26*s;gap=24*s;inner=w-38*s
        col=(inner-gap)*.44;right=left+col+gap;right_w=inner-col-gap
        top=y+h-100*s;bottom=y+43*s+footer;body_h=top-bottom
        if family=='WARP':
            top=y+h-76*s;body_h=top-bottom
            col=min(body_h,inner)
            ab=(left+(inner-col)/2,bottom+(body_h-col)/2,col,col)
            right_w=min(160*s,col*.45)
            global1=(ab[0]+col-8*s-right_w,ab[1]+col-43*s,right_w,35*s)
            tone_width=min(136*s,col*.43);button_width=(tone_width-6*s)/4
            tone_buttons={i:(ab[0]+8*s+i*(button_width+2*s),ab[1]+col-28*s,button_width,20*s) for i in range(4)}
            tabs={};tx=x+12*s
            for key,tw in (('J',90),('C',90),('WARP',80)):
                tabs[key]=(tx,y+h-63*s,tw*s,24*s);tx+=(tw+3)*s
            return dict(family=family,tone_buttons=tone_buttons,warp_statistics=(ab[0],bottom-16*s),global1=global1,
                rect=(x,y,w,h),scale=s,header=(x,y+h-32*s,w,32*s),
                close=(x+w-28*s,y+h-27*s,22*s,22*s),
                linear_tab=(x+135*s,y+h-27*s,49*s,22*s),tone_tab=(x+186*s,y+h-27*s,49*s,22*s),color_tab=(x+237*s,y+h-27*s,49*s,22*s),
                family_tabs=tabs,plot=None,ab=ab,adaptance=ab)
        padsize=min(col,body_h*.50)
        if family in {'J','C'}:
            ab=(left+(col-padsize)/2,bottom,padsize,padsize)
            curve=(left,bottom+padsize+46*s,col,body_h-padsize-46*s)
        scope_gap=12*s;wave_w=(right_w-scope_gap)*.49
        vector_size=min(right_w-wave_w-scope_gap,body_h*.45)
        scope_h=min(140*s,body_h*.49)
        vector=(right+wave_w+scope_gap,top-vector_size,vector_size,vector_size)
        controls_top=top-scope_h-26*s
        controls_w=right_w*.64;readout_w=right_w-controls_w-10*s
        controls_x=right+readout_w+10*s
        sliders={};slider_y=controls_top
        for key,field in slider_fields(family):
            height=(29+10*(len(slider_rows(field,controls_w,s))-1))*s
            sliders[key]=(controls_x,slider_y-height,controls_w,height)
            slider_y-=height+8*s
        warp_scale=(controls_x,slider_y-37*s,controls_w,29*s)
        readout_top=controls_top
        tabs={};tx=x+12*s
        for key,tw in (('J',90),('C',90),('WARP',80)):
            tabs[key]=(tx,y+h-63*s,tw*s,24*s);tx+=(tw+3)*s
        tone_buttons={i:(left+i*col/4,top-18*s,col/4-2*s,20*s) for i in range(4)}
        return dict(tone_buttons=tone_buttons,warp_statistics=(left,bottom-16*s),warp_scale=warp_scale,rect=(x,y,w,h),scale=s,header=(x,y+h-32*s,w,32*s),
            close=(x+w-28*s,y+h-27*s,22*s,22*s),
            linear_tab=(x+135*s,y+h-27*s,49*s,22*s),tone_tab=(x+186*s,y+h-27*s,49*s,22*s),color_tab=(x+237*s,y+h-27*s,49*s,22*s),
            family_tabs=tabs,
            plot=curve,ab=ab,adaptance=ab,
            waveform=(right,top-scope_h,wave_w,scope_h),
            vectorscope=vector,
            readout=(right,bottom,readout_w,readout_top-bottom),
            driver_low=(left,top+8*s,(curve[2]-3*s)/2,20*s),driver_high=(left+(curve[2]+3*s)/2,top+8*s,(curve[2]-3*s)/2,20*s),
            **sliders)


    @dataclass(frozen=True)
    class Plot:
        rect:tuple
        xmin:float
        xmax:float
        ymin:float
        ymax:float
        def screen(self,p):
            x,y,w,h=self.rect
            return (x+(p[0]-self.xmin)/(self.xmax-self.xmin)*w,y+(p[1]-self.ymin)/(self.ymax-self.ymin)*h)
        def inside(self,p):return self.xmin<=p[0]<=self.xmax and self.ymin<=p[1]<=self.ymax
        def clip(self,a,b):
            # Liang–Barsky in data coordinates, without rescaling driver geometry.
            if not np.isfinite([*a,*b]).all():return None
            dx,dy=b[0]-a[0],b[1]-a[1];lo,hi=0.,1.
            for p,q in ((-dx,a[0]-self.xmin),(dx,self.xmax-a[0]),(-dy,a[1]-self.ymin),(dy,self.ymax-a[1])):
                if abs(p)<1e-15:
                    if q<0:return None
                elif p<0:lo=max(lo,q/p)
                else:hi=min(hi,q/p)
            if lo>hi:return None
            return self.screen((a[0]+lo*dx,a[1]+lo*dy)),self.screen((a[0]+hi*dx,a[1]+hi*dy))



    @dataclass(frozen=True)
    class ABPlot(Plot):
        """Semantic coordinates stay (a,b); screen coordinates are (-b,a)."""
        def screen(self,p):
            return super().screen((-p[1],p[0]))
        def inside(self,p):return super().inside((-p[1],p[0]))
        def unscreen(self,p):
            x,y,w,h=self.rect
            return (self.ymin+(p[1]-y)/h*(self.ymax-self.ymin),-(self.xmin+(p[0]-x)/w*(self.xmax-self.xmin)))
        def clip(self,a,b):
            plain=Plot(self.rect,self.xmin,self.xmax,self.ymin,self.ymax)
            return plain.clip((-a[1],a[0]),(-b[1],b[0]))

    def square(rect):
        x,y,w,h=rect;d=min(w,h)
        return (x+(w-d)/2,y+(h-d)/2,d,d)


    def stems(family,driver):
        prefix={'WARP':'adaptance'}.get(family,family.lower())
        return (prefix+'_a',prefix+'_b')


    def plots(box,model,family,tone,driver):
        stem=family.lower()+'_'+driver
        if family in {'J','C'}:
            heights=model.heights(stem)
            reference=family.lower()+'_low'
            lim=max(ch.META[f'{reference}_{tone}'][4],float(np.max(abs(heights))))*1.06
            # Both chroma curves must remain visible, including the inactive one.
            if family in {'J','C'}:
                other=family.lower()+'_'+('high' if driver=='low' else 'low')
                lim=max(lim,float(np.max(abs(model.heights(other))))*1.06)
            curve=Plot(box['plot'],0,model.tone.plot_max*100,-lim,lim)
        else:curve=None
        ss=stems(family,driver)
        # Stable per-family physical units, independent of tone and statistics.
        lim=max(controls.meta(f'{v}_{i}')[4] for v in ss for i in range(4))*1.06
        ab=ABPlot(square(box['adaptance' if family=='WARP' else 'ab']),-lim,lim,-lim,lim)
        return curve,ab


    def pick_dot(point,positions,active,radius):
        hits=[i for i,p in enumerate(positions) if math.dist(point,p)<=radius]
        if len(hits)==1:return hits[0]
        return min(hits,key=lambda i:(abs(i-active),math.dist(point,positions[i]),i)) if hits else None


    def waveform_tone(point,box,model):
        x,y,w,h=box['waveform'];s=box['scale']
        if not x-14*s<=point[0]<=x+w:return None
        choices=[(abs(point[1]-(y+H/(model.tone.plot_max*100)*h)),i) for i,H in enumerate(model.anchors)]
        distance,index=min(choices)
        return index if distance<=7*s else None


    def curve_tone(point,plot,model,scale):
        """Shared clickable S/M/W/H tick and label strip."""
        if plot is None:return None
        x,y,w,h=plot.rect
        if not y-16*scale<=point[1]<=y+5*scale:return None
        choices=[(abs(point[0]-plot.screen((H,0))[0]),i) for i,H in enumerate(model.anchors)
                 if plot.xmin<=H<=plot.xmax]
        if not choices:return None
        distance,index=min(choices)
        return index if distance<=8*scale else None


    def curve_indicator(point,plot,model,scale):
        """Full-height tone line; callers resolve handle dots before this hit."""
        if plot is None or not in_rect(point,plot.rect):return None
        choices=[(abs(point[0]-plot.screen((H,0))[0]),i) for i,H in enumerate(model.anchors)
                 if plot.xmin<=H<=plot.xmax]
        if not choices:return None
        distance,index=min(choices)
        return index if distance<=4*scale else None


    def drag_values(params,fields,dx,dy,plot,box,slider=False,accumulator=None):
        values={}
        for i,field in enumerate(fields):
            meta=controls.meta(field);lo,hi=meta[3:5]
            old=getattr(params,field) if accumulator is None else accumulator[field]
            if slider and field.startswith('web_scale_'):
                # Display span is two stops; half-speed relative drag can continue
                # past either end without changing the stored +/-8 stop limits.
                delta=dy/box[3]
            elif slider:delta=dx*(hi-lo)/box[2]*(1 if meta[5]==0 else FINE_GAIN)
            elif len(fields)==2:
                delta=(dy if i==0 else -dx)*(plot.ymax-plot.ymin if i==0 else plot.xmax-plot.xmin)/(plot.rect[3] if i==0 else plot.rect[2])*FINE_GAIN
            else:delta=dy*(plot.ymax-plot.ymin)/plot.rect[3]*FINE_GAIN
            value=max(lo,min(hi,old+delta))
            if accumulator is not None:accumulator[field]=value
            values[field]=controls.snap(field,value)
        return values

    def _set_export_slider_fields(value):
        nonlocal slider_fields
        slider_fields = value

    def _set_export_slider_rows(value):
        nonlocal slider_rows
        slider_rows = value

    def _set_export_layout(value):
        nonlocal layout
        layout = value

    def _set_export_Plot(value):
        nonlocal Plot
        Plot = value

    def _set_export_ABPlot(value):
        nonlocal ABPlot
        ABPlot = value

    def _set_export_square(value):
        nonlocal square
        square = value

    def _set_export_stems(value):
        nonlocal stems
        stems = value

    def _set_export_plots(value):
        nonlocal plots
        plots = value

    def _set_export_pick_dot(value):
        nonlocal pick_dot
        pick_dot = value

    def _set_export_waveform_tone(value):
        nonlocal waveform_tone
        waveform_tone = value

    def _set_export_curve_tone(value):
        nonlocal curve_tone
        curve_tone = value

    def _set_export_curve_indicator(value):
        nonlocal curve_indicator
        curve_indicator = value

    def _set_export_drag_values(value):
        nonlocal drag_values
        drag_values = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ABPlot': lambda: ABPlot,
        'FINE_GAIN': lambda: FINE_GAIN,
        'Plot': lambda: Plot,
        'ch': lambda: ch,
        'controls': lambda: controls,
        'curve_indicator': lambda: curve_indicator,
        'curve_tone': lambda: curve_tone,
        'dataclass': lambda: dataclass,
        'drag_values': lambda: drag_values,
        'in_rect': lambda: in_rect,
        'layout': lambda: layout,
        'math': lambda: math,
        'np': lambda: np,
        'pick_dot': lambda: pick_dot,
        'plots': lambda: plots,
        'slider_fields': lambda: slider_fields,
        'slider_rows': lambda: slider_rows,
        'square': lambda: square,
        'stems': lambda: stems,
        'waveform_tone': lambda: waveform_tone,
    }, writers={
    'slider_fields': _set_export_slider_fields, 'slider_rows': _set_export_slider_rows, 'layout': _set_export_layout, 'Plot': _set_export_Plot, 'ABPlot': _set_export_ABPlot, 'square': _set_export_square, 'stems': _set_export_stems, 'plots': _set_export_plots, 'pick_dot': _set_export_pick_dot, 'waveform_tone': _set_export_waveform_tone, 'curve_tone': _set_export_curve_tone, 'curve_indicator': _set_export_curve_indicator, 'drag_values': _set_export_drag_values})
