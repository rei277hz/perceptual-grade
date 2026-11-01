"""H–K guarded Warp geometry on a bounded saturation display chart.

The display evaluates the page operation followed by H–K repair, retaining each
source sample's reference energy. Its centered square view follows the geometry; no image
sampling, persisted view state, or LUT bake occurs.
"""

from ..integration.components import Component

def create(_application):
    from functools import lru_cache
    retain = _application.component('ownership').retain; resolve = _application.component('ownership').resolve
    import math
    import numpy as np
    from ...color_core.modcam16 import State, hk_energy
    from ...perceptual_grade_core.model.tonal_web import rec709_contour
    from ...perceptual_grade_core.model.saturation import bounded_project
    M = _application.component('localization.i18n').M; display = _application.component('localization.i18n').display
    theme = _application.component('ui.theme')
    ABPlot = _application.component('ui.color_geometry').ABPlot; Plot = _application.component('ui.color_geometry').Plot; square = _application.component('ui.color_geometry').square

    NAMES=('Center','R','Y','G','C','B','M')
    COLORS=((.9,.9,.9,1),(.95,.3,.25,1),(.95,.8,.2,1),(.3,.8,.4,1),(.2,.8,.85,1),(.35,.5,1,1),(.85,.35,.85,1))
    HEADER=49.  # Reserved screen-space strip for tone and half-saturation controls.
    SCALE_DISPLAY=(-1.,1.)
    _rail_hovers={}
    _rail_drags={}
    _dot_drags={}


    def fields(tone,handle,scale=True):
        if handle<0:return tuple(f'adaptance_{axis}_{tone}' for axis in 'ab')
        return tuple(f'web_{kind}_{tone}_{handle}' for kind in (('chroma','hue','scale') if scale else ('chroma','hue')))


    @lru_cache(maxsize=24)
    def source(height,reference):
        contour=rec709_contour(height,reference,samples_per_edge=32)
        curves=[]
        for fraction in (.25,.5,.75,1.):
            q=contour.q.copy();q[:,1:]*=fraction
            curves.append(State(q,hk_energy(q[:,0],np.hypot(q[:,1],q[:,2]))))
        return tuple(curves)


    @lru_cache(maxsize=24)
    def geometry(model,tone):
        height=float(model.web.geometry.anchors[tone])
        contours=source(height,float(model.tone.reference))
        originals=model.web.geometry.q[tone]
        q=np.concatenate((np.array([[height,0.,0.]]),originals))
        e=np.concatenate(([height*height],model.web.geometry.energy[tone]))
        before=State(q,e);after=model.guarded_warp_targets(tone)
        spokes=[]
        for point in originals:
            q=np.broadcast_to(point,(25,3)).copy();q[:,1:]*=np.linspace(0,1,25)[:,None]
            state=State(q,hk_energy(q[:,0],np.hypot(q[:,1],q[:,2])))
            spokes.append((state,model.guarded_warp_state(state,height)))
        return dict(before=before,after=after,contours=tuple((c,model.guarded_warp_state(c,height)) for c in contours),spokes=spokes)


    @lru_cache(maxsize=24)
    def view_radius(model,tone):
        """Fit the full finite disk and controls; sample clouds never change it."""
        g=geometry(model,tone);parts=[bounded_project(g['before'].q,g['before'].energy)[1:],bounded_project(g['after'].q,g['after'].energy)]
        for _,after in (*g['contours'],*g['spokes']):
            parts.append(bounded_project(after.q,after.energy))
        points=np.concatenate(parts,axis=0)
        points=points[np.isfinite(points).all(axis=-1)]
        extent=float(np.max(np.abs(points))) if points.size else 0.
        # The centered/square view cannot pan down. Expand symmetrically to leave
        # at least 20% of its top edge clear, including in the smallest UI layout.
        upper=float(np.max(points[:,0])) if points.size else 0.
        # Include the disk even when all samples/handles are near neutral.
        return max(1.12,extent*1.12,upper/.6)


    def plot(rect,ui,model,tone,radius=None):
        """Use the complete square canvas with equal symmetric limits around neutral."""
        if radius is None:radius=view_radius(model,tone)
        return ABPlot(square(rect),-radius,radius,-radius,radius)


    def hit(ui,model,plot,point,scale):
        positions=model.guarded_saturation_targets(ui.active_tone)
        candidates=[(i-1,math.dist(point,plot.screen(q))) for i,q in enumerate(positions)
                    if handle_visible(plot,q,scale) and math.dist(point,plot.screen(q))<=8*scale]
        if not candidates:return None
        handle=min(candidates,key=lambda item:(item[0]!=ui.active_handle,item[1]))[0]
        return dict(kind='warp_dot',tone=ui.active_tone,handle=handle,fields=fields(ui.active_tone,handle),
                    plot=plot,on_handle=True,target=np.array(positions[handle+1],float),
                    disabled=False,scale=scale)


    def handle_visible(plot,point,s):
        """Never paint or pick a marker underneath the in-pad control strip."""
        return plot.inside(point) and plot.screen(point)[1]<=plot.rect[1]+plot.rect[3]-(HEADER+9)*s


    @lru_cache(maxsize=24)
    def scale_rects(model,tone,plot,s,handle):
        """Place the selected corner's single short rail beside its marker."""
        if not 0<=handle<6:return {}
        positions=model.guarded_saturation_targets(tone)[1:]
        x,y,w,h=plot.rect;rw=3*s;rh=min(80*s,h*.24);margin=5*s
        points=[plot.screen(p) if np.isfinite(p).all() else None for p in positions]
        p=points[handle]
        if p is None:return {}
        def overlap(a,b):
            return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
        candidates=[]
        side=1 if p[0]>=x+w/2 else -1
        for dx in (side*16,-side*22,side*30,-side*36,side*44,-side*50):
            for dy in (0.,-rh*.65,rh*.65):
                rect=(max(x+margin,min(p[0]+dx*s,x+w-rw-margin)),
                      max(y+margin,min(p[1]-rh/2+dy,y+h-HEADER*s-rh)),rw,rh)
                score=math.dist((rect[0]+rw/2,rect[1]+rh/2),p)
                score+=sum(overlap((rect[0]-3*s,rect[1],rw+6*s,rh),
                                   (q[0]-9*s,q[1]-9*s,18*s,18*s))*100
                           for q in points if q is not None)
                candidates.append((score,rect))
        return {handle:min(candidates,key=lambda c:c[0])[1]}


    def hit_scale(ui,model,plot,point,s):
        in_rect = _application.component('ui.geometry').in_rect
        for handle,rect in scale_rects(model,ui.active_tone,plot,s,ui.active_handle).items():
            x,y,w,h=rect
            if in_rect(point,(x-3*s,y-3*s,w+6*s,h+6*s)):
                return dict(kind='slider',fields=(f'web_scale_{ui.active_tone}_{handle}',),
                            tone=ui.active_tone,handle=handle,rect=rect,plot=None,vertical=True)
        return None


    def draw_scales(plot,model,ui,s,editor=None,node=None):
        """Draw one neutral rail; endpoint pinning is display-only."""
        d = _application.component('ui.diagram')
        rects=scale_rects(model,ui.active_tone,plot,s,ui.active_handle)
        token=_rail_drags.get(editor)
        if token is not None and _matching(token[1],node,ui.active_tone) and token[2] is not None:
            rects={token[1][3]:token[2]}
        if not rects:return
        for handle,(x,y,w,h) in rects.items():
            field=f'web_scale_{ui.active_tone}_{handle}'
            value=getattr(model.params,field);lo,hi=SCALE_DISPLAY
            yy=y+h*max(0.,min(1.,(value-lo)/(hi-lo)));zy=y+h*(-lo)/(hi-lo)
            cx=x+w/2
            d.lines(((cx,y),(cx,y+h)),theme.MUTED,1.*s)
            d.lines(((cx-3*s,zy),(cx+3*s,zy)),theme.MUTED,.7*s)
            d.lines(((cx-4*s,yy),(cx+4*s,yy)),theme.TEXT,2*s)
            message=f'{value:+.2f}';tw=len(message)*5*s+6*s
            px=max(plot.rect[0]+2*s,min(cx-tw/2,plot.rect[0]+plot.rect[2]-tw-2*s))
            py=y+h+3*s
            if py+13*s>plot.rect[1]+plot.rect[3]-HEADER*s:py=y-16*s
            py=max(plot.rect[1]+2*s,py)
            d.text(message,px+3*s,py+3*s,8*s,theme.TEXT)


    def drag(model,action,dx,dy):
        move = _application.component('ui.warp_drag').move
        return move(model,action,dx,dy)


    def paths(state,closed=False):
        """Do not connect through invalid values or across energy branches."""
        points=bounded_project(state.q,state.energy);energy=np.asarray(state.energy)
        if closed:
            points=np.r_[points,points[:1]];energy=np.r_[energy,energy[:1]]
        valid=np.isfinite(points).all(axis=-1);negative=energy<0
        start=0
        for i in range(len(points)):
            if not valid[i] or (i>start and negative[i]!=negative[i-1]):
                if i-start>1:yield points[start:i],bool(negative[start])
                start=i if valid[i] else i+1
        if len(points)-start>1:yield points[start:],bool(negative[start])


    def _screen_lines(plot,segments,color,width):
        """Clip strokes with room for their rasterized width, without GPU state."""
        d = _application.component('ui.diagram')
        x,y,w,h=plot.rect;pad=width/2+1
        rect=(x+pad,y+pad,w-2*pad,h-2*pad)
        clip=Plot(rect,rect[0],rect[0]+rect[2],rect[1],rect[1]+rect[3])
        coords=[]
        for a,b in zip(segments[::2],segments[1::2]):
            clipped=clip.clip(a,b)
            if clipped is not None:coords.extend(clipped)
        d.lines(coords,color,width)


    def _circle(plot,point,radius,color,hollow=False,width=1):
        """Intersect the small convex marker polygon with the screen rectangle."""
        d = _application.component('ui.diagram')
        points=[(point[0]+radius*math.cos(i*math.tau/32),point[1]+radius*math.sin(i*math.tau/32)) for i in range(32)]
        if hollow:
            _screen_lines(plot,[q for a,b in zip(points,points[1:]+points[:1]) for q in (a,b)],color,width)
            return
        x,y,w,h=plot.rect
        for axis,bound,sign in ((0,x,1),(0,x+w,-1),(1,y,1),(1,y+h,-1)):
            clipped=[]
            for a,b in zip(points,points[1:]+points[:1]):
                ai=sign*(a[axis]-bound)>=0;bi=sign*(b[axis]-bound)>=0
                if ai:clipped.append(a)
                if ai!=bi:
                    t=(bound-a[axis])/(b[axis]-a[axis])
                    clipped.append(tuple(a[j]+t*(b[j]-a[j]) for j in (0,1)))
            points=clipped
            if not points:return
        d.triangles([q for i in range(1,len(points)-1) for q in (points[0],points[i],points[i+1])],color)


    def _text(plot,label,position,size,color):
        import blf
        d = _application.component('ui.diagram')
        x,y,w,h=plot.rect;blf.size(0,int(size));tw,th=blf.dimensions(0,display(label))
        if tw>w-4 or th>h-4:return
        px=max(x+2,min(position[0],x+w-tw-2));py=max(y+2,min(position[1],y+h-th-2))
        d.text(label,px,py,size,color)


    def target(plot,position,color,s):
        """Resolve-style open square corners around a fixed Rec.709 reference."""
        if not handle_visible(plot,position,s):return
        x,y=plot.screen(position);radius=6*s;length=2.5*s;segments=[]
        for horizontal in (-1,1):
            for vertical in (-1,1):
                corner=(x+horizontal*radius,y+vertical*radius)
                segments.extend(((corner[0]-horizontal*length,corner[1]),corner,
                    corner,(corner[0],corner[1]-vertical*length)))
        _screen_lines(plot,segments,(*color[:3],.8),.8*s)


    def draw(plot,model,ui,s,samples=None,editor=None,node=None):
        d = _application.component('ui.diagram')
        g=geometry(model,ui.active_tone);x,y,w,h=plot.rect
        d.rectangle(plot.rect,theme.PLOT)
        for fraction in np.linspace(0,1,5):
            xx=x+w*fraction;yy=y+h*fraction
            d.lines(((xx,y),(xx,y+h),(x,yy),(x+w,yy)),(*theme.GRID[:3],.35),.6*s)
        zero=plot.screen((0.,0.))
        if plot.xmin<=0<=plot.xmax:d.lines(((zero[0],y),(zero[0],y+h)),(*theme.OUTLINE[:3],.8),.8*s)
        if plot.ymin<=0<=plot.ymax:d.lines(((x,zero[1]),(x+w,zero[1])),(*theme.OUTLINE[:3],.8),.8*s)
        if samples is not None:samples(plot)
        radius=active_rim_radius(editor,node,model,ui.active_tone,ui.active_handle)
        if radius is not None:
            ring=np.linspace(0,math.tau,257)
            d.curve(plot,radius*np.column_stack((np.cos(ring),np.sin(ring))),(*theme.MUTED[:3],.75),.9*s)
        for index,(_,after) in enumerate(g['contours']):
            opacity=.8 if index==len(g['contours'])-1 else .4
            for points,negative in paths(after,True):
                d.curve(plot,points,(*theme.ORANGE[:3],opacity),1.*s,negative)
        for _,after in g['spokes']:
            for points,negative in paths(after):d.curve(plot,points,(*theme.ORANGE[:3],.4),.6*s,negative)
        for index,(position,color) in enumerate(zip(bounded_project(g['before'].q,g['before'].energy)[1:],COLORS[1:]),1):
            target(plot,position,color,s)
            if handle_visible(plot,position,s):
                pos=plot.screen(position)
                label_y=min(pos[1]+7*s,y+h-(HEADER+10)*s)
                _text(plot,NAMES[index],(pos[0]+8*s,label_y),7*s,color)
        markers=list(enumerate(zip(bounded_project(g['after'].q,g['after'].energy),COLORS)))
        # A dark guarded corner may overlap neutral or another corner; retain the
        # selected handle's visible marker, matching active-first hit testing.
        markers.sort(key=lambda item:item[0]==ui.active_handle+1)
        for index,(after,color) in markers:
            if handle_visible(plot,after,s):
                pos=plot.screen(after);selected=index==ui.active_handle+1
                _circle(plot,pos,(4 if selected else 3)*s,color,g['after'].energy[index]<0)
                if selected:_circle(plot,pos,6*s,theme.TEXT,True,.8*s)
        # Narrow squares put the in-pad tone row over the usual +a label location.
        a_top=y+h-54*s
        for label,pos in ((M('symbol.b_minus'),(x+w-15*s,y+h/2+5*s)),(M('symbol.b_plus'),(x+2*s,y+h/2+5*s)),(M('symbol.a_plus'),(x+w/2+4*s,a_top)),(M('symbol.a_minus'),(x+w/2+4*s,y+3*s))):_text(plot,label,pos,8*s,theme.TEXT)
        caption=M('diagram.warp.chroma_stops') if radius is not None else M('diagram.warp.saturation')
        _text(plot,caption,(x+4*s,y+4*s),7*s,theme.MUTED)
        return g


    def _rail_state(node,tone,handle):
        return (retain(node,'warp.rail'),node.as_pointer(),int(tone),int(handle))


    def _matching(state,node,tone):
        if state is None or node is None:return False
        try:return resolve(state[0]).as_pointer()==node.as_pointer()==state[1] and state[2]==int(tone)
        except (AttributeError,ReferenceError,RuntimeError):return False


    def set_rail_hover(editor,node,tone,handle=None):
        """Transient editor state only; never select a handle or write Grade RNA."""
        if editor is None:return False
        departed=[other for other in _rail_hovers if other[0]==editor[0] and other!=editor]
        for other in departed:_rail_hovers.pop(other,None)
        old=_rail_hovers.get(editor)
        if handle is None:
            _rail_hovers.pop(editor,None)
            return bool(departed) or old is not None
        state=_rail_state(node,tone,handle)
        if _matching(old,node,tone) and old[3]==state[3]:return bool(departed)
        _rail_hovers[editor]=state
        return True


    def begin_scale_drag(editor,node,tone,handle,rect=None):
        if editor is None:return None
        token=(editor,_rail_state(node,tone,handle),rect)
        _rail_drags[editor]=token
        return token


    def end_scale_drag(token):
        if token is not None and _rail_drags.get(token[0]) is token:
            _rail_drags.pop(token[0],None)


    def begin_drag(editor,node,action):
        """Expose only the active gesture's contextual rim to its own editor."""
        if editor is None:return None
        token=(editor,_rail_state(node,action['tone'],action['handle']),action)
        _dot_drags[editor]=token
        return token


    def end_drag(token):
        if token is not None and _dot_drags.get(token[0]) is token:
            _dot_drags.pop(token[0],None)


    def active_rim_radius(editor,node,model,tone,handle):
        """The raw E=0 threshold is local to the crossing handle, never permanent."""
        token=_dot_drags.get(editor)
        if token is None or not _matching(token[1],node,tone) or token[1][3]!=handle or handle<0:
            return None
        action=token[2]
        if not action.get('rim_active',False):return None
        radius=action.get('rim_radius')
        if radius is None:radius=model.guard_rim_radius(tone,handle)
        return float(radius) if math.isfinite(radius) and radius>0 else None


    def readout_handle(editor,node,tone,selected):
        token=_rail_drags.get(editor)
        if token is not None and _matching(token[1],node,tone):return token[1][3]
        state=_rail_hovers.get(editor)
        if _matching(state,node,tone):return state[3]
        return selected


    def clear_rail_hovers(window=None):
        """Forget pointer state on departure from a window or on teardown."""
        if window is None:
            changed=bool(_rail_hovers);_rail_hovers.clear();return changed
        key=window.as_pointer()
        keys=[editor for editor in _rail_hovers if editor[0]==key]
        for editor in keys:_rail_hovers.pop(editor,None)
        return bool(keys)


    def prune_rails(live_editors,valid):
        for cache in (_rail_hovers,_rail_drags,_dot_drags):
            for editor,value in tuple(cache.items()):
                state=value if cache is _rail_hovers else value[1]
                try:
                    owner=resolve(state[0])
                    keep=owner is not None and editor in live_editors and valid(owner)
                except (ReferenceError,RuntimeError):keep=False
                if not keep:cache.pop(editor,None)


    def clear():
        """Release all CPU/GPU geometry and transient pointers on lifecycle reset."""
        _rail_hovers.clear();_rail_drags.clear();_dot_drags.clear()
        for function in (source,geometry,view_radius,scale_rects):
            clear_cache=getattr(function,'cache_clear',None)
            if clear_cache is not None:clear_cache()

    def _set_export_COLORS(value):
        nonlocal COLORS
        COLORS = value

    def _set_export_HEADER(value):
        nonlocal HEADER
        HEADER = value

    def _set_export_NAMES(value):
        nonlocal NAMES
        NAMES = value

    def _set_export_SCALE_DISPLAY(value):
        nonlocal SCALE_DISPLAY
        SCALE_DISPLAY = value

    def _set_export__dot_drags(value):
        nonlocal _dot_drags
        _dot_drags = value

    def _set_export__rail_drags(value):
        nonlocal _rail_drags
        _rail_drags = value

    def _set_export__rail_hovers(value):
        nonlocal _rail_hovers
        _rail_hovers = value

    def _set_export_fields(value):
        nonlocal fields
        fields = value

    def _set_export_source(value):
        nonlocal source
        source = value

    def _set_export_geometry(value):
        nonlocal geometry
        geometry = value

    def _set_export_view_radius(value):
        nonlocal view_radius
        view_radius = value

    def _set_export_plot(value):
        nonlocal plot
        plot = value

    def _set_export_hit(value):
        nonlocal hit
        hit = value

    def _set_export_handle_visible(value):
        nonlocal handle_visible
        handle_visible = value

    def _set_export_scale_rects(value):
        nonlocal scale_rects
        scale_rects = value

    def _set_export_hit_scale(value):
        nonlocal hit_scale
        hit_scale = value

    def _set_export_draw_scales(value):
        nonlocal draw_scales
        draw_scales = value

    def _set_export_drag(value):
        nonlocal drag
        drag = value

    def _set_export_paths(value):
        nonlocal paths
        paths = value

    def _set_export__screen_lines(value):
        nonlocal _screen_lines
        _screen_lines = value

    def _set_export__circle(value):
        nonlocal _circle
        _circle = value

    def _set_export__text(value):
        nonlocal _text
        _text = value

    def _set_export_target(value):
        nonlocal target
        target = value

    def _set_export_draw(value):
        nonlocal draw
        draw = value

    def _set_export__rail_state(value):
        nonlocal _rail_state
        _rail_state = value

    def _set_export__matching(value):
        nonlocal _matching
        _matching = value

    def _set_export_set_rail_hover(value):
        nonlocal set_rail_hover
        set_rail_hover = value

    def _set_export_begin_scale_drag(value):
        nonlocal begin_scale_drag
        begin_scale_drag = value

    def _set_export_end_scale_drag(value):
        nonlocal end_scale_drag
        end_scale_drag = value

    def _set_export_begin_drag(value):
        nonlocal begin_drag
        begin_drag = value

    def _set_export_end_drag(value):
        nonlocal end_drag
        end_drag = value

    def _set_export_active_rim_radius(value):
        nonlocal active_rim_radius
        active_rim_radius = value

    def _set_export_readout_handle(value):
        nonlocal readout_handle
        readout_handle = value

    def _set_export_clear_rail_hovers(value):
        nonlocal clear_rail_hovers
        clear_rail_hovers = value

    def _set_export_prune_rails(value):
        nonlocal prune_rails
        prune_rails = value

    def _set_export_clear(value):
        nonlocal clear
        clear = value

    return Component(readers={
        "__name__": lambda: __name__,
        'ABPlot': lambda: ABPlot,
        'COLORS': lambda: COLORS,
        'HEADER': lambda: HEADER,
        'M': lambda: M,
        'NAMES': lambda: NAMES,
        'Plot': lambda: Plot,
        'SCALE_DISPLAY': lambda: SCALE_DISPLAY,
        'State': lambda: State,
        '_circle': lambda: _circle,
        '_dot_drags': lambda: _dot_drags,
        '_matching': lambda: _matching,
        '_rail_drags': lambda: _rail_drags,
        '_rail_hovers': lambda: _rail_hovers,
        '_rail_state': lambda: _rail_state,
        '_screen_lines': lambda: _screen_lines,
        '_text': lambda: _text,
        'active_rim_radius': lambda: active_rim_radius,
        'begin_drag': lambda: begin_drag,
        'begin_scale_drag': lambda: begin_scale_drag,
        'bounded_project': lambda: bounded_project,
        'clear': lambda: clear,
        'clear_rail_hovers': lambda: clear_rail_hovers,
        'display': lambda: display,
        'drag': lambda: drag,
        'draw': lambda: draw,
        'draw_scales': lambda: draw_scales,
        'end_drag': lambda: end_drag,
        'end_scale_drag': lambda: end_scale_drag,
        'fields': lambda: fields,
        'geometry': lambda: geometry,
        'handle_visible': lambda: handle_visible,
        'hit': lambda: hit,
        'hit_scale': lambda: hit_scale,
        'hk_energy': lambda: hk_energy,
        'lru_cache': lambda: lru_cache,
        'math': lambda: math,
        'np': lambda: np,
        'paths': lambda: paths,
        'plot': lambda: plot,
        'prune_rails': lambda: prune_rails,
        'readout_handle': lambda: readout_handle,
        'rec709_contour': lambda: rec709_contour,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'scale_rects': lambda: scale_rects,
        'set_rail_hover': lambda: set_rail_hover,
        'source': lambda: source,
        'square': lambda: square,
        'target': lambda: target,
        'theme': lambda: theme,
        'view_radius': lambda: view_radius,
    }, writers={
        'COLORS': _set_export_COLORS,
        'HEADER': _set_export_HEADER,
        'NAMES': _set_export_NAMES,
        'SCALE_DISPLAY': _set_export_SCALE_DISPLAY,
        '_dot_drags': _set_export__dot_drags,
        '_rail_drags': _set_export__rail_drags,
        '_rail_hovers': _set_export__rail_hovers,
    'fields': _set_export_fields, 'source': _set_export_source, 'geometry': _set_export_geometry, 'view_radius': _set_export_view_radius, 'plot': _set_export_plot, 'hit': _set_export_hit, 'handle_visible': _set_export_handle_visible, 'scale_rects': _set_export_scale_rects, 'hit_scale': _set_export_hit_scale, 'draw_scales': _set_export_draw_scales, 'drag': _set_export_drag, 'paths': _set_export_paths, '_screen_lines': _set_export__screen_lines, '_circle': _set_export__circle, '_text': _set_export__text, 'target': _set_export_target, 'draw': _set_export_draw, '_rail_state': _set_export__rail_state, '_matching': _set_export__matching, 'set_rail_hover': _set_export_set_rail_hover, 'begin_scale_drag': _set_export_begin_scale_drag, 'end_scale_drag': _set_export_end_scale_drag, 'begin_drag': _set_export_begin_drag, 'end_drag': _set_export_end_drag, 'active_rim_radius': _set_export_active_rim_radius, 'readout_handle': _set_export_readout_handle, 'clear_rail_hovers': _set_export_clear_rail_hovers, 'prune_rails': _set_export_prune_rails, 'clear': _set_export_clear})
