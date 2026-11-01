"""Compact Stage 3 contribution inspector: dot editing, no tonal number boxes."""

from ..integration.components import Component

def create(_application):
    M = _application.component('localization.i18n').M
    from ...perceptual_grade_core.localization.controls import control_readout
    marker = _application.component('localization.i18n').marker

    import numpy as np
    nodes = _application.component('graph.nodes')
    theme = _application.component('ui.theme')
    from ...perceptual_grade_core.model import color as ch
    cs = _application.component('scopes.color')
    geo = _application.component('ui.color_geometry')
    from ...perceptual_grade_core.model import statistics as robust
    controls = _application.component('model.controls')
    in_rect = _application.component('ui.geometry').in_rect

    LOW=theme.ADAPTED
    HIGH=theme.ORANGE
    FAMILY_TITLES={'J':M('family.j.label'),'C':M('family.c.label'),'WARP':M('family.warp.label')}
    _DENSITY={}


    def scope_retry_rect(rect,s):
        x,y,w,h=rect
        return (x+4*s,y+h-20*s,min(w-8*s,112*s),17*s)


    def gpu_status_rects(rect,s):
        x,y,w,h=rect
        return {'gpu_details':(x+4*s,y+h-57*s,56*s,17*s),
                'gpu_retry':(x+64*s,y+h-57*s,min(w-68*s,80*s),17*s)}


    def gpu_action(context,action,role='gpu'):
        """Keep full diagnostics available without retaining any node RNA."""
        gpu = _application.component('graph.gpu')
        tr = _application.component('localization.i18n').tr; wrap = _application.component('localization.i18n').wrap
        if action=='gpu_retry':
            gpu.retry(role=role)
            return
        diagnostic=gpu.status(role=role)['diagnostic']
        if diagnostic is None:return
        context.window_manager.clipboard=gpu.details(role=role)
        rows=[tr('gpu.details.copied'),
              tr('gpu.details.device',backend=diagnostic.backend,device=diagnostic.device),
              tr('gpu.details.attempt',attempt=diagnostic.attempt,generation=diagnostic.generation),
              tr('gpu.details.build',fingerprint=diagnostic.fingerprint)]
        rows.extend(wrap(diagnostic.reason,90))
        rows=tuple(rows)
        def draw(self,_context):
            for row in rows:self.layout.label(text=row)
        context.window_manager.popup_menu(draw,title=tr('gpu.details.title'),icon='INFO')


    def draw_scope_status(node,rect,s,kind='full',role='gpu'):
        """Keep retained measurements visible while clearly reporting freshness."""
        jobs = _application.component('scopes.jobs')
        d = _application.component('ui.diagram')
        state=jobs.status(node,kind)
        inputs=_application.component('scopes.sampling').scope_input_status(node)
        native=inputs.get('native',False)
        label={'preparing':M('scope.status.preparing'),'updating':M('scope.status.updating'),
               'error':M('scope.status.error')}.get(state['state'])
        if state['state']=='paused':label=M('scope.status.paused')
        elif state['state'] in {'waiting','capturing','stale'}:label=M('scope.status.updating')
        if state['state']!='paused' and native and not _application.component('graph.output_guard').available():
            label=M('scope.status.waiting_native')
        frames=tuple(dict.fromkeys(state.get('frames',())))
        scene_frame=state.get('scene_frame')
        if frames or scene_frame is not None:
            tr=_application.component('localization.i18n').tr
            frame_label=tr('scope.status.frame' if state['state']=='current' else 'scope.status.previous_frame',
                           frames=str(scene_frame) if scene_frame is not None else ', '.join(map(str,frames)))
            if frames and frames!=(scene_frame,):
                frame_label+=' · '+tr('scope.status.source_frame',frames=', '.join(map(str,frames)))
            label=(tr(label)+' · ' if label is not None else '')+frame_label
        x,y,w,h=rect
        if label is not None:
            if state['state']=='error':
                d.rectangle(scope_retry_rect(rect,s),theme.HEADER)
            d.text(label,x+7*s,y+h-14*s,7*s,theme.ORANGE if state['state']=='error' else theme.MUTED)
        error=inputs.get('error') or state.get('error')
        if error:
            wrap = _application.component('localization.i18n').wrap
            issue = _application.component('localization.i18n').error_text(error)
            rows = wrap(issue,max(20,int((w-14*s)/(4*s))))[:3]
            for index,row in enumerate(rows):
                d.text(row,x+7*s,y+(8+10*(len(rows)-index-1))*s,7*s,theme.ORANGE)
        gpu = _application.component('graph.gpu')
        health=gpu.status(role=role)
        if health['state']=='ready':return
        gpu_rect=rect
        d.text(M('gpu.status.probing') if health['state']=='probing' else M('gpu.status.fallback'),
               x+7*s,gpu_rect[1]+gpu_rect[3]-32*s,7*s,theme.ORANGE)
        labels={'gpu_details':M('gpu.details'),'gpu_retry':M('gpu.retry')}
        for action,button in gpu_status_rects(gpu_rect,s).items():
            if action=='gpu_retry' and health['state']=='probing':continue
            if action=='gpu_details' and health['diagnostic'] is None:continue
            d.rectangle(button,theme.HEADER)
            d.text(labels[action],button[0]+3*s,button[1]+5*s,7*s,theme.TEXT)


    def control_state(node,space,box):
        """Handle geometry only. Hover/picking must never sample an image."""
        p=nodes.read_chromatic(node);m=nodes.color_model(node,space);ui=node.pg_chromatic
        if ui.family=='WARP':
            warp = _application.component('ui.warp'); warp_view = _application.component('ui.warp_view')
            radius=warp_view.radius(box.get('_warp_editor'),node,ui.active_tone)
            curve=None;ab=warp.plot(box['ab'],ui,m,ui.active_tone,radius=radius)
        else:curve,ab=geo.plots(box,m,ui.family,ui.active_tone,ui.active_chroma)
        return p,m,curve,ab


    def state(node,space,box):
        p,m,curve,ab=control_state(node,space,box)
        result=None;error=''
        try:result=cs.get(node,space)
        except (ValueError,RuntimeError,ReferenceError,TypeError) as exc:error=str(exc)
        return p,m,curve,ab,result,error


    def sliders(family):
        return geo.slider_fields(family)


    def hit(node,space,box,point):
        """Pick both chroma curves, resolve overlaps by selection then tone distance."""
        p,m,curve,ab=control_state(node,space,box);ui=node.pg_chromatic;s=box['scale']
        for family,rect in box['family_tabs'].items():
            if in_rect(point,rect):return dict(kind='family',value=family)
        if ui.family=='WARP':
            warp = _application.component('ui.warp')
            for tone,rect in box['tone_buttons'].items():
                if in_rect(point,rect):return dict(kind='tone',value=tone,source='button')
            if in_rect(point,box['global1']):
                return dict(kind='slider',fields=('adaptance_half_saturation',),rect=box['global1'],plot=None)
            action=warp.hit_scale(ui,m,ab,point,s)
            if action is not None:return action
            action=warp.hit(ui,m,ab,point,s)
            if action is not None:return action
        for key,field in sliders(ui.family):
            if in_rect(point,box[key]):return dict(kind='slider',fields=(field,),rect=box[key],plot=None)
        if ui.family in {'J','C'}:
            for key,driver in (('driver_low','low'),('driver_high','high')):
                if in_rect(point,box[key]):return dict(kind='driver',value=driver,source='selector')
        if ui.family!='WARP':
            tone=geo.waveform_tone(point,box,m)
            if tone is not None:return dict(kind='tone',value=tone,source='waveform')
        if ui.family=='WARP':return dict(kind='consume')
        if ab and in_rect(point,ab.rect):
            stems=geo.stems(ui.family,ui.active_chroma)
            positions=[ab.screen(v) for v in zip(*(m.heights(stem) for stem in stems))]
            tone=geo.pick_dot(point,positions,ui.active_tone,7*s)
            on_handle=tone is not None
            if tone is None:tone=ui.active_tone
            return dict(kind='dot',tone=tone,fields=tuple(f'{stem}_{tone}' for stem in stems),plot=ab,on_handle=on_handle)
        if curve:
            drivers=(ui.active_chroma,'high' if ui.active_chroma=='low' else 'low') if ui.family in {'J','C'} else (None,)
            for driver in drivers:
                stem=ui.family.lower()+'_'+driver
                positions=[curve.screen(v) for v in zip(m.anchors,m.heights(stem))]
                tone=geo.pick_dot(point,positions,ui.active_tone,7*s)
                if tone is not None:
                    return dict(kind='dot',tone=tone,driver=driver,fields=(f'{stem}_{tone}',),plot=curve,on_handle=True)
            tone=geo.curve_tone(point,curve,m,s)
            if tone is not None:return dict(kind='tone',value=tone,source='tick')
            tone=geo.curve_indicator(point,curve,m,s)
            if tone is not None:
                if ui.family in {'J','C'} and tone==ui.active_tone:
                    return dict(kind='driver',value='high' if ui.active_chroma=='low' else 'low',source='indicator',tone=tone)
                return dict(kind='tone',value=tone,source='indicator')
            if in_rect(point,curve.rect):
                tone=ui.active_tone;driver=drivers[0]
                stem=ui.family.lower()+'_'+driver
                return dict(kind='dot',tone=tone,driver=driver,fields=(f'{stem}_{tone}',),plot=curve,on_handle=False)
        return dict(kind='consume')


    def apply_click(node,action):
        ui=node.pg_chromatic;kind=action['kind']
        if kind=='family':ui.family=action['value']
        elif kind=='driver':ui.active_chroma=action['value']
        elif kind=='tone':ui.active_tone=action['value']
        elif kind=='warp_dot':ui.active_tone=action['tone'];ui.active_handle=action['handle']
        elif kind=='slider' and 'handle' in action:ui.active_handle=action['handle']
        elif kind=='dot':
            ui.active_tone=action['tone']
            if action.get('driver') is not None:ui.active_chroma=action['driver']


    def draw_slider(rect,field,p,s):
        d = _application.component('ui.diagram')
        meta=controls.meta(field);value=getattr(p,field);lo,hi=meta[3:5]
        x,y,w,h=rect;message=f'{value:.{meta[5]}f}'
        rows=geo.slider_rows(field,w,s)
        for i,row in enumerate(rows):d.text(row,x,y+(19+10*(len(rows)-1-i))*s,9*s)
        d.text(message,x+w-len(message)*5.5*s,y+19*s,9*s,theme.ORANGE)
        yy=y+5*s;end=(x+max(0.,min(1.,(value-lo)/(hi-lo)))*w,yy)
        d.lines([(x,yy),(x+w,yy)],theme.OUTLINE,3*s)
        d.lines([(x,yy),end],theme.BLUE,3*s);d.circle(end,3*s,theme.ORANGE)


    def axes(plot,s,xlabel,ylabel,polar=False,result=None,neutral=False,tone_anchors=None,active_tone=None):
        d = _application.component('ui.diagram'); pad_background = _application.component('ui.pad_background')
        x,y,w,h=plot.rect;d.rectangle(plot.rect,theme.PLOT)
        if polar:pad_background.draw(plot,result if result is not None and 'sectors' in result else None,neutral)
        for t in (0,.5,1):
            xx=x+t*w;yy=y+t*h
            grid=[(x,yy),(x+w,yy)]
            if tone_anchors is None:grid=[(xx,y),(xx,y+h)]+grid
            d.lines(grid,(*theme.GRID[:3],.30 if polar else .55),.6*s)
            if not polar:
                xv=plot.xmin+t*(plot.xmax-plot.xmin);yv=plot.ymin+t*(plot.ymax-plot.ymin)
                if tone_anchors is None:d.text(f'{xv:.0f}',xx-(12 if t==1 else 3)*s,y-12*s,7*s,theme.MUTED)
                d.text(f'{yv:.1f}' if abs(yv)<5 else f'{yv:.0f}',x-25*s,yy-3*s,7*s,theme.MUTED)
        if tone_anchors is not None:
            for i,(label,value) in enumerate(zip(ch.IDS,tone_anchors)):
                if not plot.xmin<=value<=plot.xmax:continue
                xx=plot.screen((value,0))[0]
                color=theme.ORANGE if i==active_tone else theme.MUTED
                d.lines([(xx,y),(xx,y+h)],(*theme.GRID[:3],.95 if i==active_tone else .55),(.9 if i==active_tone else .6)*s)
                d.lines([(xx,y-3*s),(xx,y)],color,.7*s)
                d.text(marker('tone',label),min(x+w-6*s,max(x,xx-3*s)),y-12*s,7*s,color)
        ox,oy=plot.screen((0,0))
        if plot.xmin<=0<=plot.xmax:d.lines([(ox,y),(ox,y+h)],(*theme.OUTLINE[:3],.75),s)
        if plot.ymin<=0<=plot.ymax:d.lines([(x,oy),(x+w,oy)],(*theme.OUTLINE[:3],.75),s)
        if xlabel:d.text(xlabel,x,y-(12 if polar else 24)*s,8*s,theme.MUTED)
        if ylabel:d.text(ylabel,x,y+h+9*s,9*s)
        if polar:
            for r in (.5,1):d.circle((x+w/2,y+h/2),w*.44*r,(*theme.GRID[:3],.30),True,.7*s)
            for label,pos in [(M('symbol.b_minus'),(x+w-15*s,y+h/2+5*s)),(M('symbol.b_plus'),(x+2*s,y+h/2+5*s)),(M('symbol.a_plus'),(x+w/2+4*s,y+h-10*s)),(M('symbol.a_minus'),(x+w/2+4*s,y+3*s))]:
                d.text(label,*pos,8*s,theme.TEXT)


    def points(plot,positions,tone,s,color,dim=False):
        d = _application.component('ui.diagram')
        for i,p in enumerate(positions):
            if not plot.inside(p):continue
            xy=plot.screen(p);col=(*color[:3],.20 if dim else 1.)
            d.circle(xy,(3.6 if i==tone else 2.5)*s,col,hollow=dim or i!=tone,width=s)
            if i==tone and not dim:d.circle(xy,5.1*s,theme.TEXT,True,.8*s)


    def histograms(plot,result,drivers,selected,s):
        d = _application.component('ui.diagram')
        if result is None:return
        if '_active_tone' in result:
            warp_samples = _application.component('ui.warp_samples')
            warp_samples.draw_histograms(plot,result,drivers,selected)
            return
        x,y,w,h=plot.rect
        all_values=[result['histograms'][name] for name in drivers]
        maximum=max(.01,max(float(np.max(abs(v))) for v in all_values))
        # Separate right-hand coverage scale; primary y remains coefficient units.
        for name,values in sorted(zip(drivers,all_values),key=lambda item:item[0]==selected):
            color=LOW if name=='low' else HIGH if name=='high' else theme.ADAPTED
            alpha=.26 if name==selected or name=='tone' else .055
            coords=[];n=len(values);dx=w/n
            for i,value in enumerate(values):
                xx=x+i*dx;yy=y+float(value)/maximum*h*.78
                coords.extend(((xx,y),(xx+dx,y),(xx+dx,yy),(xx,y),(xx+dx,yy),(xx,yy)))
            d.triangles(coords,(*color[:3],alpha))


    def draw_curve(plot,model,stem,tone,color,s,dim=False):
        d = _application.component('ui.diagram')
        H=np.linspace(plot.xmin,plot.xmax,300);curve=model.curve(H,stem)
        d.curve(plot,list(zip(H,curve)),(*color[:3],.18 if dim else 1),1.7*s if not dim else s)
        positions=list(zip(model.anchors,model.heights(stem)))
        if not dim:
            for a,b in zip(positions,model.curve(model.anchors,stem)):
                d.curve(plot,[a,(a[0],b)],(*color[:3],.45),.7*s,True)
        points(plot,positions,tone,s,color,dim)




    def neutral_stats(plot,stats,s,caption=True):
        d = _application.component('ui.diagram')
        from ...perceptual_grade_core.model.saturation import bounded_from_linear
        pixel_to_unit=(plot.xmax-plot.xmin)/plot.rect[2]
        # Statistical membership and its fivefold magnification retain linear
        # saturation meaning; only the displayed geometry is compactified.
        raw=5*robust.ellipse(stats,min_radius=0.)
        pts=bounded_from_linear(raw)
        projected_center=bounded_from_linear(5*np.asarray(stats['center']))
        # Keep the former three-pixel visibility floor for a rank-deficient fit,
        # after the nonlinear display projection rather than in the statistics.
        delta=pts-projected_center
        lengths=np.linalg.norm(delta,axis=-1)
        floor=3*s*pixel_to_unit
        fallback=np.column_stack((np.cos(np.linspace(0,2*np.pi,len(pts),endpoint=False)),
                                  np.sin(np.linspace(0,2*np.pi,len(pts),endpoint=False))))
        directions=np.divide(delta,lengths[:,None],out=fallback.copy(),where=lengths[:,None]>0)
        pts=projected_center+directions*np.maximum(lengths,floor)[:,None]
        pts=np.r_[pts,pts[:1]]
        d.curve(plot,pts,theme.ADAPTED,1.3*s)
        if plot.inside(projected_center):
            radius=5*s*pixel_to_unit
            d.curve(plot,[projected_center+(-radius,0),projected_center+(radius,0)],theme.TEXT,1.5*s)
            d.curve(plot,[projected_center+(0,-radius),projected_center+(0,radius)],theme.TEXT,1.5*s)
        if not caption:return
        x,y,w,h=plot.rect
        wrap = _application.component('localization.i18n').wrap
        for i,line in enumerate(wrap(M('diagram.adaptance.statistics'),max(12,int(w/(4*s))))):
            d.text(line,x,y-(15+10*i)*s,8*s,theme.MUTED)


    def density_levels(values):
        maximum=max(1.,float(np.max(values)))
        # A vanishing tail must not become a visible dot just because ceil(x)>0.
        return np.clip(np.rint(np.log1p(np.maximum(values,0))/np.log1p(maximum)*7),0,7).astype(int)


    def density(rect,values,color,alpha=1):
        d = _application.component('ui.diagram')
        key=(tuple(rect),id(values))
        if key not in _DENSITY:
            rows,cols=values.shape;levels=density_levels(values)
            x,y,w,h=rect;dx=w/cols;dy=h/rows;groups=[]
            corners=np.array([(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)])*[dx,dy]
            for level in range(1,8):
                ys,xs=np.where(levels==level)
                origin=np.stack([x+xs*dx,y+ys*dy],axis=-1)
                groups.append(d.CachedTriangles((origin[:,None,:]+corners).reshape(-1,2).tolist()))
            if len(_DENSITY)>=12:_DENSITY.clear()
            _DENSITY[key]=(values,groups)
        for level,coords in enumerate(_DENSITY[key][1],1):
            d.triangles(coords,(*color[:3],alpha*(.12+.88*level/7)))


    def draw_scopes(box,model,tone,result,external=None,warp=False):
        d = _application.component('ui.diagram')
        if warp:return
        s=box['scale']
        if not warp:
            rect=box['waveform'];x,y,w,h=rect;maximum=model.tone.plot_max*100
            d.rectangle(rect,theme.PLOT)
            if result:
                if 'waveform_levels' in result:
                    draw_density = _application.component('ui.warp_samples').draw_density
                    draw_density(rect,result['waveform_levels'],theme.ADAPTED,.90)
                else:density(rect,result['waveform_selected'],theme.ADAPTED,.90)
            if external:
                if 'waveform_levels' in external:
                    draw_density = _application.component('ui.warp_samples').draw_density
                    draw_density(rect,external['waveform_levels'],theme.COMPARISON,.95)
                else:density(rect,external['waveform'],theme.COMPARISON,.95)
            d.text(M('scope.waveform'),x,y+h+10*s,8*s)
            for i,H in enumerate(model.anchors):
                yy=y+H/maximum*h;color=theme.ORANGE if i==tone else theme.REFERENCE
                d.lines([(x,yy),(x+w,yy)],(*color[:3],1 if i==tone else .45),s)
                d.text(marker('tone',ch.IDS[i]),x-10*s,yy-3*s,8*s,color)
        draw_vectorscope(box['vectorscope'],
            ({'levels':result['vectorscope_levels']} if 'vectorscope_levels' in result else result['vectorscope']) if result else None,
            ({'levels':external['vectorscope_levels']} if 'vectorscope_levels' in external else external['vectorscope']) if external else None,s)


    def draw_vectorscope(rect,current=None,external=None,s=1,title=M('scope.vectorscope')):
        d = _application.component('ui.diagram')
        rect=geo.square(rect);x,y,w,h=rect;d.rectangle(rect,theme.PLOT)
        for values,color in ((current,theme.ADAPTED),(external,theme.COMPARISON)):
            if values is None:continue
            if 'levels' in values:
                draw_density = _application.component('ui.warp_samples').draw_density
                draw_density(rect,values['levels'],color)
            else:density(rect,values['density'],color)
        center=(x+w/2,y+h/2)
        for r in (.25,.5):d.circle(center,w*r/.65/2,theme.GRID,True,.7*s)
        d.lines([(x,center[1]),(x+w,center[1]),(center[0],y),(center[0],y+h)],theme.GRID,.7*s)
        # 100% Rec.709 color-bar target positions in Cb/Cr signal coordinates.
        for name,rgb,color in [('R',(1,0,0),theme.COLORS['cr'][0]),('Y',(1,1,0),theme.COLORS['yb'][1]),('G',(0,1,0),(.4,.8,.5,1)),('C',(0,1,1),theme.COLORS['cr'][1]),('B',(0,0,1),theme.COLORS['yb'][0]),('M',(1,0,1),(.8,.45,.8,1))]:
            yy=np.dot(rgb,[.2126,.7152,.0722]);cb=(rgb[2]-yy)/1.8556;cr=(rgb[0]-yy)/1.5748
            p=(center[0]+cb/1.3*w,center[1]+cr/1.3*h);d.circle(p,1.8*s,color,True,.7*s);d.text(marker('scope',name),p[0]+2*s,p[1]+2*s,6*s,color)
        d.text(title,x,y+h+10*s,8*s)


    def draw_board(context,node,space,params,box):
        d = _application.component('ui.diagram')
        p,m,curve,ab,result,error=state(node,space,box);ui=node.pg_chromatic;tone=ui.active_tone;s=box['scale'];x,y,w,h=box['rect']
        d.rectangle(box['rect'],theme.PANEL);d.rectangle(box['header'],theme.HEADER);d.draw_header(box,'CHROMATIC',node)
        for family,rect in box['family_tabs'].items():
            d.rectangle(rect,theme.BLUE if family==ui.family else theme.HEADER)
            d.text(FAMILY_TITLES[family],rect[0]+5*s,rect[1]+8*s,9*s,theme.TEXT if family==ui.family else theme.MUTED)
        if ui.family!='WARP':
            for key,field in sliders(ui.family):draw_slider(box[key],field,p,s)
        comparison = _application.component('scopes.reference')
        external=comparison.selected(node,result['_base']) if result else None
        draw_scopes(box,m,tone,result,external,ui.family=='WARP')
        clouds = _application.component('scopes.clouds')
        draw_scope_status(node,box['ab'] if ui.family=='WARP' else box['waveform'],s,
                          role='scope_gpu' if clouds.enabled() else 'gpu')
        if ui.family=='WARP':
            warp = _application.component('ui.warp'); warp_samples = _application.component('ui.warp_samples')
            warp.draw(ab,m,ui,s,samples=lambda plot:warp_samples.draw(plot,result,external,s),
                      editor=box.get('_warp_editor'),node=node)
            stats=result.get('neutral') if result else None
            if stats is not None:
                neutral_stats(ab,stats,s,False)
            warp.draw_scales(ab,m,ui,s,editor=box.get('_warp_editor'),node=node)
            sx,sy=box['warp_statistics']
            d.text(M('warp.statistics'),sx,sy,7*s,theme.MUTED)
            for index,rect in box['tone_buttons'].items():
                d.rectangle(rect,theme.BLUE if index==tone else theme.HEADER)
                d.text(marker('tone',ch.IDS[index]),rect[0]+rect[2]/2-3*s,rect[1]+6*s,9*s)
            # Overlay controls last so the pad and sample cloud cannot cover them.
            d.rectangle(box['global1'],theme.PLOT)
            draw_slider(box['global1'],'adaptance_half_saturation',p,s)
            d.draw_footer(node,box)
            return
        elif ui.family in {'J','C'}:
            target=ui.family;stem=target.lower()
            axes(curve,s,M('diagram.j.units') if target=='J' else M('diagram.c.units'),'',tone_anchors=m.anchors,active_tone=tone)
            histograms(curve,result,('low','high'),ui.active_chroma,s)
            for driver,color,key,label in [('low',LOW,'driver_low','Cₗ'),('high',HIGH,'driver_high','Cₕ')]:
                r=box[key];selected=driver==ui.active_chroma
                d.rectangle(r,theme.HEADER);d.text(M(f'readout.{target.lower()}_{driver}'),r[0]+3*s,r[1]+7*s,8*s,(*color[:3],1 if selected else .35))
                if selected:d.lines([(r[0],r[1]),(r[0]+r[2],r[1])],color,2*s)
            for driver in ('high','low') if ui.active_chroma=='low' else ('low','high'):
                draw_curve(curve,m,stem+'_'+driver,tone,LOW if driver=='low' else HIGH,s,driver!=ui.active_chroma)
            axes(ab,s,M(f'diagram.{stem}.ab_units'),M(f'diagram.{stem}.ab_label'),True,result)
            H=np.linspace(0,m.tone.plot_max*100,250)
            d.curve(ab,list(zip(m.curve(H,stem+'_a'),m.curve(H,stem+'_b'))),theme.ORANGE,1.5*s)
            points(ab,list(zip(m.heights(stem+'_a'),m.heights(stem+'_b'))),tone,s,theme.ORANGE)
        # Below the waveform: values on the left, compact sliders on the right.
        rx,ry,rw,rh=box['readout'];line_y=ry+rh-8*s
        names=(f'{ui.family.lower()}_{ui.active_chroma}_{tone}',f'{ui.family.lower()}_a_{tone}',f'{ui.family.lower()}_b_{tone}')
        for i,name in enumerate(names):
            label=M(control_readout(name.rsplit('_',1)[0]))
            d.text(label,rx,line_y-i*24*s,7*s,theme.MUTED)
            d.text(f'{getattr(p,name):+.2f}',rx,line_y-(i*24+11)*s,8*s,theme.TEXT)
        d.draw_footer(node,box)

    def _set_export_FAMILY_TITLES(value):
        nonlocal FAMILY_TITLES
        FAMILY_TITLES = value

    def _set_export_HIGH(value):
        nonlocal HIGH
        HIGH = value

    def _set_export_LOW(value):
        nonlocal LOW
        LOW = value

    def _set_export__DENSITY(value):
        nonlocal _DENSITY
        _DENSITY = value

    def _set_export_scope_retry_rect(value):
        nonlocal scope_retry_rect
        scope_retry_rect = value

    def _set_export_gpu_status_rects(value):
        nonlocal gpu_status_rects
        gpu_status_rects = value

    def _set_export_gpu_action(value):
        nonlocal gpu_action
        gpu_action = value

    def _set_export_draw_scope_status(value):
        nonlocal draw_scope_status
        draw_scope_status = value

    def _set_export_control_state(value):
        nonlocal control_state
        control_state = value

    def _set_export_state(value):
        nonlocal state
        state = value

    def _set_export_sliders(value):
        nonlocal sliders
        sliders = value

    def _set_export_hit(value):
        nonlocal hit
        hit = value

    def _set_export_apply_click(value):
        nonlocal apply_click
        apply_click = value

    def _set_export_draw_slider(value):
        nonlocal draw_slider
        draw_slider = value

    def _set_export_axes(value):
        nonlocal axes
        axes = value

    def _set_export_points(value):
        nonlocal points
        points = value

    def _set_export_histograms(value):
        nonlocal histograms
        histograms = value

    def _set_export_draw_curve(value):
        nonlocal draw_curve
        draw_curve = value

    def _set_export_neutral_stats(value):
        nonlocal neutral_stats
        neutral_stats = value

    def _set_export_density_levels(value):
        nonlocal density_levels
        density_levels = value

    def _set_export_density(value):
        nonlocal density
        density = value

    def _set_export_draw_scopes(value):
        nonlocal draw_scopes
        draw_scopes = value

    def _set_export_draw_vectorscope(value):
        nonlocal draw_vectorscope
        draw_vectorscope = value

    def _set_export_draw_board(value):
        nonlocal draw_board
        draw_board = value

    return Component(readers={
        "__name__": lambda: __name__,
        'FAMILY_TITLES': lambda: FAMILY_TITLES,
        'HIGH': lambda: HIGH,
        'LOW': lambda: LOW,
        'M': lambda: M,
        '_DENSITY': lambda: _DENSITY,
        'apply_click': lambda: apply_click,
        'axes': lambda: axes,
        'ch': lambda: ch,
        'control_readout': lambda: control_readout,
        'control_state': lambda: control_state,
        'controls': lambda: controls,
        'cs': lambda: cs,
        'density': lambda: density,
        'density_levels': lambda: density_levels,
        'draw_board': lambda: draw_board,
        'draw_curve': lambda: draw_curve,
        'draw_scope_status': lambda: draw_scope_status,
        'draw_scopes': lambda: draw_scopes,
        'draw_slider': lambda: draw_slider,
        'draw_vectorscope': lambda: draw_vectorscope,
        'geo': lambda: geo,
        'gpu_action': lambda: gpu_action,
        'gpu_status_rects': lambda: gpu_status_rects,
        'histograms': lambda: histograms,
        'hit': lambda: hit,
        'in_rect': lambda: in_rect,
        'marker': lambda: marker,
        'neutral_stats': lambda: neutral_stats,
        'nodes': lambda: nodes,
        'np': lambda: np,
        'points': lambda: points,
        'robust': lambda: robust,
        'scope_retry_rect': lambda: scope_retry_rect,
        'sliders': lambda: sliders,
        'state': lambda: state,
        'theme': lambda: theme,
    }, writers={
        'FAMILY_TITLES': _set_export_FAMILY_TITLES,
        'HIGH': _set_export_HIGH,
        'LOW': _set_export_LOW,
        '_DENSITY': _set_export__DENSITY,
    'scope_retry_rect': _set_export_scope_retry_rect, 'gpu_status_rects': _set_export_gpu_status_rects, 'gpu_action': _set_export_gpu_action, 'draw_scope_status': _set_export_draw_scope_status, 'control_state': _set_export_control_state, 'state': _set_export_state, 'sliders': _set_export_sliders, 'hit': _set_export_hit, 'apply_click': _set_export_apply_click, 'draw_slider': _set_export_draw_slider, 'axes': _set_export_axes, 'points': _set_export_points, 'histograms': _set_export_histograms, 'draw_curve': _set_export_draw_curve, 'neutral_stats': _set_export_neutral_stats, 'density_levels': _set_export_density_levels, 'density': _set_export_density, 'draw_scopes': _set_export_draw_scopes, 'draw_vectorscope': _set_export_draw_vectorscope, 'draw_board': _set_export_draw_board})
