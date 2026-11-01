"""Stage 2's actual overlay drawing; all data comes from scopes.py/model."""

from ..integration.components import Component

def create(_application):
    M = _application.component('localization.i18n').M
    tr = _application.component('localization.i18n').tr; marker = _application.component('localization.i18n').marker

    import math
    import numpy as np
    nodes = _application.component('graph.nodes')
    from ...perceptual_grade_core.model import tone as tm
    scopes = _application.component('scopes.sampling')
    theme = _application.component('ui.theme')
    editing = _application.component('graph.editing')
    Plot = _application.component('ui.color_geometry').Plot


    def fmt(value):
        return f'{value:,.0f}' if value>=100 else (f'{value:.2f}' if value>=1 else f'{value:.3f}')


    def level_y(j,rect,maximum):
        return rect[1]+rect[3]*max(0,min(1,j/maximum))


    def hit_level(point,box,model,side='before'):
        x,y=point;rx,ry,rw,rh=box[side];s=box['scale']
        if not rx-12*s<=x<=rx+rw+3*s:return None
        values=model.source if side=='before' else model.target
        choices=[(abs(y-level_y(v,box[side],model.plot_max)),i) for i,v in enumerate(values)]
        distance,index=min(choices)
        return index if distance<=7*s else None


    def curve_plot(box,model):
        return Plot(box['chart'].rect,0,model.plot_max,0,model.plot_max)


    def hit_curve(point,box,model,active):
        plot=curve_plot(box,model);hits=[]
        for i,(src,dst) in enumerate(zip(model.source,model.target)):
            if not 0<=src<=model.plot_max:continue
            for out in (dst,float(model.remap(src))):
                if 0<=out<=model.plot_max and math.dist(point,plot.screen((src,out)))<=7*box['scale']:
                    hits.append(i);break
        return hits[0] if len(hits)==1 else active if active in hits else None


    def density(rect,data,color=theme.ADAPTED):
        d = _application.component('ui.diagram')
        if data is None:return
        bins=data['density'];rows,cols=bins.shape
        # Identical density normalization on before and after; counts aren't
        # independently stretched to make the two images look equally populated.
        levels=np.minimum(7,np.floor(np.log2(1+bins)*2)).astype(int)
        rx,ry,rw,rh=rect;dx=rw/cols;dy=rh/rows
        for level in range(1,8):
            ys,xs=np.where(levels==level);coords=[]
            for i,j in zip(xs,ys):
                x=rx+i*dx;y=ry+j*dy
                coords.extend(((x,y),(x+dx,y),(x+dx,y+dy),(x,y),(x+dx,y+dy),(x,y+dy)))
            d.triangles(coords,(*color[:3],.22+.1*level))


    def dashed_connector(a,b,color,s):
        d = _application.component('ui.diagram')
        dx=b[0]-a[0];points=[]
        for i in range(49):
            t=i/48;u=1-t
            points.append((u**3*a[0]+3*u*u*t*(a[0]+dx*.48)+3*u*t*t*(b[0]-dx*.48)+t**3*b[0],
                           (u**3+3*u*u*t)*a[1]+(3*u*t*t+t**3)*b[1]))
        length=0.;coords=[]
        for p,q in zip(points,points[1:]):
            if int(length/(4*s))%2==0:coords.extend((p,q))
            length+=math.dist(p,q)
        d.lines(coords,color,s)


    def draw_board(context,node,space,params,box):
        d = _application.component('ui.diagram')
        s=box['scale'];x,y,w,h=box['rect'];tone=nodes.read_tone(node);space=nodes.grading_space(node);model=tm.model(tone,space)
        selected=getattr(context.window_manager,'pg_selected_tone',3)
        try:result=scopes.get_tone(node,space,ui=True)
        except (ValueError,RuntimeError,ReferenceError,TypeError):result=None
        comparison = _application.component('scopes.reference')
        external=comparison.get(node,result) if result else None
        stack=editing.is_target(node)
        d.rectangle(box['rect'],theme.PANEL);d.rectangle(box['header'],theme.HEADER);d.draw_header(box,'REMAP',node)
        d.draw_exposure(params,box)
        px,py,pw,ph=box['peak'];d.rectangle(box['peak'],theme.HEADER)
        d.text(tr('peak.diagram.label',value=model.peak),px+6*s,py+9*s,10*s)
        for key,title in (('before',M('scope.stack.input' if stack else 'scope.before')),
                          ('after',M('scope.stack.output' if stack else 'scope.after'))):
            rect=box[key];rx,ry,rw,rh=rect;d.rectangle(rect,theme.PLOT)
            density(rect,result[key] if result else None)
            if key=='after' and external:density(rect,external['distribution'],theme.COMPARISON)
            for j in np.arange(0,model.plot_max,.5):
                yy=level_y(j,rect,model.plot_max)
                d.lines([(rx,yy),(rx+rw,yy)],(*theme.GRID[:3],.55),.6*s)
            d.text(title,rx,ry+rh+9*s,10*s)
            # Targets are declared landmarks, never the soft curve's actual output.
            if key=='before':label=tr('diagram.input_j.value',j=model.source[selected]);nits=model.j_to_nits(model.source[selected])
            else:label=tr('diagram.target_j.value',tone=tr('tone.'+tm.LEVEL_NAMES[selected].lower()),j=model.target[selected]);nits=model.target_nits[selected]
            d.text(label,rx-3*s,ry+rh+39*s,8*s,theme.ORANGE)
            d.text(tr('unit.neutral_nits.format',nits=fmt(float(nits))),rx-3*s,ry+rh+27*s,8*s,theme.MUTED)
            if stack:d.text(M('scope.stack.selected_reference'),rx+4*s,ry+8*s,7*s,theme.MUTED)
        for i,(src,dest) in enumerate(zip(model.source,model.target)):
            left=box['before'];right=box['after'];sy=level_y(src,left,model.plot_max);dy=level_y(dest,right,model.plot_max)
            color=theme.ORANGE if i==selected else theme.REFERENCE
            if src<=model.plot_max:d.lines([(left[0],sy),(left[0]+left[2],sy)],color,s)
            d.text(marker('tone',tm.LEVEL_NAMES[i][0])+('↑' if src>model.plot_max else ''),left[0]-11*s,sy-3*s,8*s,color)
            d.circle((left[0]+left[2],sy),2.5*s,color)
            d.lines([(right[0],dy),(right[0]+right[2],dy)],color,s)
            d.circle((right[0],dy),2.5*s,color,True,s)
            d.text(marker('tone',tm.LEVEL_NAMES[i][0]),right[0]+right[2]+4*s,dy-3*s,8*s,color)
            dashed_connector((left[0]+left[2]+3*s,sy),(right[0]-3*s,dy),color,s)
        draw_scope_status = _application.component('ui.color').draw_scope_status
        draw_scope_status(node,box['before'],s)
        plot=curve_plot(box,model);cx,cy,cw,ch=plot.rect
        d.text(M('diagram.grade_curve' if stack else 'diagram.tone_curve'),cx,cy+ch+39*s,10*s)
        d.rectangle(plot.rect,theme.PLOT)
        if result:
            histograms=[(result['after']['histogram'],(*theme.MUTED[:3],.24))]
            if external:histograms.append((external['distribution']['histogram'],(*theme.COMPARISON[:3],.40)))
            for bins,color in histograms:
                # Each signal owns its vertical scale. A tone edit can concentrate
                # the primary bins while the unchanged reference stays in place.
                maximum=max(1,float(np.max(bins)))
                coords=[]
                for i,v in enumerate(bins):
                    xx=cx+i/len(bins)*cw;yy=cy+v/maximum*ch*.72;dx=cw/len(bins)
                    coords.extend(((xx,cy),(xx+dx,cy),(xx+dx,yy),(xx,cy),(xx+dx,yy),(xx,yy)))
                d.triangles(coords,color)
        for tick in (0,1,2,model.plot_max):
            if tick>model.plot_max:continue
            xx,yy=plot.screen((tick,tick))
            d.lines([(xx,cy),(xx,cy+ch),(cx,yy),(cx+cw,yy)],theme.GRID,.7*s)
            label=f'{tick:.2f}' if tick==model.plot_max else f'{tick:g}'
            d.text(label,xx-(18 if tick==model.plot_max else 3)*s,cy-13*s,7*s,theme.MUTED)
        d.curve(plot,[(0,0),(model.plot_max,model.plot_max)],theme.REFERENCE,s,True)
        values=np.linspace(0,model.plot_max,300)
        d.curve(plot,list(zip(values,model.remap(values))),theme.ORANGE,1.6*s)
        for i,(src,dest) in enumerate(zip(model.source,model.target)):
            if 0<=src<=model.plot_max:
                color=theme.ORANGE if i==selected else theme.REFERENCE
                d.circle(plot.screen((src,dest)),2.4*s,color,True,s)
                out=float(model.remap(src))
                if 0<=out<=model.plot_max:d.circle(plot.screen((src,out)),(3 if i==selected else 2)*s,color)
        white=min(1/model.plot_max,1)*cw
        d.rectangle((cx,cy+ch+4*s,white,3*s),theme.REFERENCE)
        d.rectangle((cx+white,cy+ch+4*s,cw-white,3*s),theme.ADAPTED)
        d.text(M('symbol.sdr'),cx,cy+ch+12*s,7*s,theme.MUTED);d.text(M('symbol.hdr'),cx+white+3*s,cy+ch+12*s,7*s,theme.MUTED)
        d.text(M('diagram.input_j'),cx+cw/2-12*s,cy-25*s,8*s,theme.MUTED)
        d.text(M('diagram.output_j'),cx+4*s,cy+ch-12*s,8*s,theme.MUTED)
        d.draw_value_slider(box['fit'],M('control.remap_fit.label'),tone.fit,-2,6,s,0)
        d.draw_footer(node,box)

    def _set_export_fmt(value):
        nonlocal fmt
        fmt = value

    def _set_export_level_y(value):
        nonlocal level_y
        level_y = value

    def _set_export_hit_level(value):
        nonlocal hit_level
        hit_level = value

    def _set_export_curve_plot(value):
        nonlocal curve_plot
        curve_plot = value

    def _set_export_hit_curve(value):
        nonlocal hit_curve
        hit_curve = value

    def _set_export_density(value):
        nonlocal density
        density = value

    def _set_export_dashed_connector(value):
        nonlocal dashed_connector
        dashed_connector = value

    def _set_export_draw_board(value):
        nonlocal draw_board
        draw_board = value

    return Component(readers={
        "__name__": lambda: __name__,
        'M': lambda: M,
        'Plot': lambda: Plot,
        'curve_plot': lambda: curve_plot,
        'dashed_connector': lambda: dashed_connector,
        'density': lambda: density,
        'draw_board': lambda: draw_board,
        'editing': lambda: editing,
        'fmt': lambda: fmt,
        'hit_curve': lambda: hit_curve,
        'hit_level': lambda: hit_level,
        'level_y': lambda: level_y,
        'marker': lambda: marker,
        'math': lambda: math,
        'nodes': lambda: nodes,
        'np': lambda: np,
        'scopes': lambda: scopes,
        'theme': lambda: theme,
        'tm': lambda: tm,
        'tr': lambda: tr,
    }, writers={
    'fmt': _set_export_fmt, 'level_y': _set_export_level_y, 'hit_level': _set_export_hit_level, 'curve_plot': _set_export_curve_plot, 'hit_curve': _set_export_hit_curve, 'density': _set_export_density, 'dashed_connector': _set_export_dashed_connector, 'draw_board': _set_export_draw_board})
