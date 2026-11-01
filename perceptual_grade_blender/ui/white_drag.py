"""Source-white direction control in the displayed Rec.709 signal vectorscope.

A fixed sparse image grid defines a nominal 95% robust data region. The mean
of points inside that region defines the gesture, in vectorscope coordinates.
The region is fitted once; selected pixels and alpha weights stay fixed.
Each motion recomputes its 2x2 Jacobian through the current Stage-1 matrix and
signed transfer function. The inverse determines direction only. The xy chart
scale and DRAG_GAIN determine travel. The UI uses one quarter of a source-dot
pixel per pointer pixel in the regular interior. Mouse conversion and sensitivity
are applied together by move_pixels, before the boundary/recovery calculation.
At the safe-white boundary, movement follows the feasible tangent and stops
at the first edge. Recovery-directed gestures blend toward the working white.
Only the gesture's x/y changes; no scope statistic enters the grading model.
"""

from ..integration.components import Component

def create(_application):
    from dataclasses import replace
    import numpy as np
    from ...perceptual_grade_core.model.scope_color import project_rec709 as project
    from ...perceptual_grade_core.model import linear as cm, statistics as rs


    SAMPLE_GRID = (16, 16)
    SAMPLE_LIMIT = SAMPLE_GRID[0]*SAMPLE_GRID[1]
    # Source-white chart pixels per pointer pixel, shared by every UI caller.
    DRAG_GAIN = .25
    SCOPE_MATRIX = np.array([[-.2126, -.7152, 1-.0722],
                             [1-.2126, -.7152, -.0722]])/np.array([[1.8556], [1.5748]])
    WHITE_NORMALS = np.asarray(cm.WHITE_PLANES,dtype=float)
    WHITE_NORMALS = WHITE_NORMALS/np.linalg.norm(WHITE_NORMALS[:,:2],axis=1)[:,None]
    BOUNDARY_BLEND_WIDTH = .02
    BOUNDARY_TOLERANCE = 1e-7  # Includes the RNA float32 round trip.


    def smooth01(value):
        value=float(np.clip(value,0.,1.))
        return value*value*(3-2*value)


    def boundary_move(xy,step):
        """Project onto the active tangent cone, then stop at the first edge.

        Projecting a distant endpoint can jump between different polygon edges.
        This local projection removes the blocked component before moving and
        never renormalizes it. Both operations can only shorten the requested step.
        """
        xy=np.asarray(xy,dtype=float);step=np.asarray(step,dtype=float)
        distances=WHITE_NORMALS[:,:2]@xy+WHITE_NORMALS[:,2]
        active=WHITE_NORMALS[distances<=BOUNDARY_TOLERANCE,:2]
        if len(active) and np.any(active@step < -1e-14):
            candidates=[np.zeros(2)]
            for normal in active:
                tangent=step-normal*np.dot(normal,step)
                if np.all(active@tangent>=-1e-14):candidates.append(tangent)
            step=min(candidates,key=lambda v:np.sum((v-step)**2))
        rate=WHITE_NORMALS[:,:2]@step
        crossing=rate < -1e-14
        fraction=min(1.,float(np.min(np.maximum(distances[crossing],0.)/-rate[crossing]))) if np.any(crossing) else 1.
        # This final projection corrects roundoff at an attained edge only.
        return np.asarray(cm.safe_source_white(xy+fraction*step))


    def sparse_samples(rgb, alpha):
        """Fixed cell-center samples, selected before conversion or color math."""
        rgb = np.asarray(rgb)
        alpha = np.asarray(alpha)
        if rgb.ndim == 3:
            height, width = rgb.shape[:2]
            nx, ny = min(SAMPLE_GRID[0], width), min(SAMPLE_GRID[1], height)
            if nx == 0 or ny == 0:
                return np.empty((0, 3)), np.empty(0)
            xs = ((np.arange(nx)+.5)*width/nx).astype(int)
            ys = ((np.arange(ny)+.5)*height/ny).astype(int)
            rgb, alpha = rgb[ys[:, None], xs[None, :]], alpha[ys[:, None], xs[None, :]]
        else:
            rgb, alpha = rgb.reshape(-1, 3), alpha.ravel()
            count = min(SAMPLE_LIMIT, len(rgb))
            if count:
                ids = ((np.arange(count)+.5)*len(rgb)/count).astype(int)
                rgb, alpha = rgb[ids], alpha[ids]
        return np.asarray(rgb, dtype=float).reshape(-1, 3), np.asarray(alpha, dtype=float).ravel()


    def confidence_membership(positions, alpha):
        """Fit the 95% data contour, then retain the original in-region pixels.

        Alpha enters the fit through deterministic equal-weight resampling and
        the final mean through normalized weights. The ellipse center estimates
        robust location; the drag reference is the mean of its selected points.
        """
        fit, _ = rs.weighted_sample(positions, alpha, limit=SAMPLE_LIMIT)
        center, covariance = rs.detmcd(fit)
        distance = rs.distances(positions, center, covariance)
        keep = distance <= rs.CHI2_95
        if not np.any(keep):
            # A finite singleton still gives an attainable, deterministic target.
            keep[np.argmin(distance)] = True
        return keep




    class WhiteDrag:
        def __init__(self,rgb,alpha,params,space,scope_rgb=None):
            self.space=space
            rgb,alpha=sparse_samples(rgb,alpha)
            scope_is_input=scope_rgb is None
            scope_rgb=rgb if scope_is_input else np.asarray(scope_rgb,dtype=float).reshape(-1,3)
            if len(scope_rgb)!=len(rgb):
                raise ValueError('Scope anchor samples must match source samples')
            sample_indices=np.arange(len(rgb))
            good=np.isfinite(rgb).all(axis=-1)&np.isfinite(alpha)&(alpha>0)
            good&=np.isfinite(scope_rgb).all(axis=-1)
            rgb=rgb[good];alpha=alpha[good];sample_indices=sample_indices[good]
            scope_rgb=scope_rgb[good]
            if len(rgb)==0:
                rgb=np.full((1,3),.18);alpha=np.ones(1)
                sample_indices=np.array([0],dtype=int)
            if len(scope_rgb)==0:scope_rgb=rgb.copy()
            # Scaling alpha leaves its mean unchanged and avoids sum overflow.
            alpha=alpha/np.max(alpha)
            positions=project(scope_rgb@np.asarray(cm.matrices(params,space)['rgb']).T,space) if scope_is_input else project(scope_rgb,space)
            self.sample_count=len(rgb)
            self.inlier_mask=confidence_membership(positions,alpha)
            self.selection_indices=sample_indices[self.inlier_mask]
            self.rgb=rgb[self.inlier_mask].copy()
            weights=alpha[self.inlier_mask]
            self.weights=weights/weights.sum()
            self.initial_center=self.position(params)
            self._home_parameters=None
            self._home_scope=None
            self._home_jacobian=None

        def position(self,params):
            out=self.rgb@np.asarray(cm.matrices(params,self.space)['rgb']).T
            return self.weights@project(out,self.space)

        def reset(self,params):
            self.initial_center=self.position(params)

        def jacobian(self,params,rgb=None,weights=None):
            """Derivative before safety clipping, including boundary one-sided limits.

            CAT16 gains are destination_cone*y / (a*x+b*y+c). Differentiating
            that expression avoids finite differences across the guarded polygon.
            """
            if rgb is None:rgb,weights=self.rgb,self.weights
            x,y=cm.safe_source_white((params.source_x,params.source_y))
            cat=np.asarray(cm.CAT16)
            a=cat[:,0]-cat[:,2];b=cat[:,1]-cat[:,2]
            f=a*x+b*y+cat[:,2]
            destination=cat@np.asarray(cm.xyz(cm.white(self.space)))
            gain=destination*y/f
            derivative=np.stack((-destination*y*a/f**2,
                                  destination*(f-y*b)/f**2),axis=-1)
            cones=np.asarray(rgb)@(cat@np.asarray(cm.rgb_to_xyz(self.space))).T
            output=(np.asarray(cm.xyz_to_rgb(self.space if isinstance(self.space,cm.GradeSpace) else 'Linear Rec.709'))
                    @np.asarray(cm.directional_matrix(params,self.space))
                    @np.asarray(cm.ICAT16))*2**params.exposure
            linear=(cones*gain)@output.T
            magnitude=np.abs(linear)
            slope=np.where(magnitude<.018,4.5,
                           1.099*.45*np.maximum(magnitude,.018)**(-.55))
            return np.stack([weights@((slope*((cones*derivative[:,i])@output.T))
                                       @SCOPE_MATRIX.T) for i in range(2)],axis=-1)

        def usable_jacobian(self,params):
            """Neutral-probe orientation for dark or collapsed image responses."""
            jac=self.jacobian(params)
            u,s,vh=np.linalg.svd(jac,full_matrices=False)
            observable=s[0]>=1e-10
            if not observable:
                jac=self.jacobian(params,np.full((1,3),.18),np.ones(1))
                u,s,vh=np.linalg.svd(jac,full_matrices=False)
                if s[0]<1e-10:
                    neutral=replace(cm.neutral_parameters(self.space),
                                    source_x=params.source_x,source_y=params.source_y)
                    jac=self.jacobian(neutral,np.full((1,3),.18),np.ones(1))
                    u,s,vh=np.linalg.svd(jac,full_matrices=False)
            return jac,u,s,vh,observable

        def direction(self,params,delta):
            return self.steering(params,delta)[0]

        def steering(self,params,delta):
            """Blend the inverse direction with a frame oriented toward working white.

            The recovery frame maps a drag toward the working-white scope position
            to an xy step toward that white. Its handedness is fixed by the image
            response at working white. Near a fold or guard edge, this frame takes
            over smoothly; opposite drags remain opposite and never auto-center.
            """
            jac,u,s,vh,observable=self.usable_jacobian(params)
            weakness=1-smooth01((s[-1]/max(s[0],1e-30))/.08)
            damping=max(1e-12,s[0]*max(1e-5,.04*weakness))
            direction=vh.T@((s/(s*s+damping*damping))*(u.T@delta))
            length=np.linalg.norm(direction)
            direction=direction/length if length>=1e-15 else delta/np.linalg.norm(delta)
            xy=np.asarray(cm.safe_source_white((params.source_x,params.source_y)))
            toward=np.asarray(cm.white(self.space))-xy
            distance=np.linalg.norm(toward)
            edge_distance=float(np.min(WHITE_NORMALS[:,:2]@xy+WHITE_NORMALS[:,2]))
            risk=max(weakness,1-smooth01(edge_distance/BOUNDARY_BLEND_WIDTH))
            home=replace(params,source_x=cm.white(self.space)[0],source_y=cm.white(self.space)[1])
            if home!=self._home_parameters:
                self._home_parameters=home;self._home_scope=self.position(home)
                self._home_jacobian,_,hs,_,_=self.usable_jacobian(home)
                if hs[-1]<hs[0]*1e-8:
                    self._home_jacobian=self.jacobian(cm.neutral_parameters(self.space),np.full((1,3),.18),np.ones(1))
            handedness=1. if np.linalg.det(self._home_jacobian)>=0 else -1.
            # A determinant sign change crosses a fold relative to working white.
            # Weakness reaches one before that sign changes, preventing a branch jump.
            if np.linalg.det(jac)*handedness<0:risk=1.
            returning=0.
            if distance>1e-10:
                scope_toward=self._home_scope-self.position(params)
                if np.linalg.norm(scope_toward)<1e-12:scope_toward=self._home_jacobian@toward
                scope_length=np.linalg.norm(scope_toward)
                if scope_length>1e-15:
                    a=toward/distance;b=scope_toward/scope_length
                    perpendicular_a=np.array([-a[1],a[0]])
                    perpendicular_b=np.array([-b[1],b[0]])
                    alignment=float(np.dot(delta,b))
                    preferred=a*alignment+handedness*perpendicular_a*np.dot(delta,perpendicular_b)
                    # A locally valid inverse can still point to the other global
                    # branch. Favor the white-connected direction when they disagree.
                    conflict=1-smooth01((float(np.dot(direction,preferred))+.25)/.75)
                    risk=max(risk,conflict)
                    weight=risk*smooth01(distance/BOUNDARY_BLEND_WIDTH)
                    # Angular interpolation keeps unit speed without a cancellation
                    # at opposite vector directions. Full recovery returns exactly
                    # the same preferred frame on either side of an inverse fold.
                    angle=np.arctan2(direction[0]*preferred[1]-direction[1]*preferred[0],np.dot(direction,preferred))
                    c,sine=np.cos(weight*angle),np.sin(weight*angle)
                    direction=np.array([c*direction[0]-sine*direction[1],sine*direction[0]+c*direction[1]])
                    returning=weight*max(0.,alignment)
            return direction,returning,observable

        def move_pixels(self,params,pointer_delta,xy_per_pixel):
            """Move from a raw pointer delta and the displayed xy chart scale."""
            delta=np.asarray(pointer_delta,dtype=float)*(xy_per_pixel*DRAG_GAIN)
            return self._move_xy(params,delta)

        def _move_xy(self,params,delta):
            """Integrate a requested xy displacement after mouse sensitivity.

            Interior travel has the length of delta. Feasible tangents and a
            first-edge stop replace distant endpoint projection. A backtracking
            check prevents finite steps from oscillating across a response turn.
            """
            delta=np.asarray(delta,dtype=float)
            xy=np.asarray(cm.safe_source_white((params.source_x,params.source_y)))
            distance=np.linalg.norm(delta)
            if np.isfinite(delta).all() and distance>0:
                pointer=delta/distance
                direction,recovery,observable=self.steering(params,pointer)
                before=self.position(params)
                home=np.asarray(cm.white(self.space))
                for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
                    trial=boundary_move(xy,direction*distance*fraction)
                    if np.linalg.norm(trial-xy)<1e-12:break
                    current=replace(params,source_x=float(trial[0]),source_y=float(trial[1]))
                    progress=float(np.dot(self.position(current)-before,pointer))
                    returning=recovery>1e-6 and np.sum((trial-home)**2)<np.sum((xy-home)**2)-1e-15
                    if progress>1e-12 or returning or not observable:
                        xy=trial;break
            return dict(source_x=float(xy[0]),source_y=float(xy[1]))


    def compose_stage1(matrices,directions):
        """Compose the enabled Stack Stage-1 matrices in occurrence order.

        Matrices use the same column-vector convention as the reference model. An
        inverse occurrence contributes its matrix inverse; later entries multiply
        on the left. The inputs are expected in the shared native RGB basis used
        by the Stack kernel.
        """
        matrices=tuple(np.asarray(matrix,dtype=float) for matrix in matrices)
        directions=tuple(directions)
        if len(matrices)!=len(directions):
            raise ValueError('Stage-1 matrices and directions must have matching lengths')
        result=np.eye(3)
        for matrix,direction in zip(matrices,directions):
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
                raise ValueError('Stage-1 matrices must be finite 3 by 3 arrays')
            if direction=='INVERSE':
                try:matrix=np.linalg.inv(matrix)
                except np.linalg.LinAlgError as exc:raise ValueError('Inverse Stage-1 matrix is singular') from exc
            elif direction!='FORWARD':
                raise ValueError('Stack directions must be FORWARD or INVERSE')
            result=matrix@result
        if not np.isfinite(result).all():
            raise ValueError('Composed Stage-1 matrix is not finite')
        return result


    class StackWhiteDrag(WhiteDrag):
        """Source-white steering against a whole-Stack scope snapshot.

        Candidate Stage-1 matrices are evaluated through a private shallow copy of
        the prepared production Stack evaluator. Its immutable nonlinear kernels,
        inverse solvers and tone tables are shared, while per-occurrence matrices
        and endpoints are replaced on the copy. The captured LUT output anchors
        the transient preview, so model/LUT interpolation error cannot move its
        starting point.
        """
        def __init__(self,rgb,alpha,params,space,matrices,directions,edited_occurrences,
                     authoritative_matrix,native_left,native_right,to_native,from_native,
                     stack_model=None,input_rgb=None,prepared=None):
            # Exact candidate evaluation is prepared before the base constructor so
            # it can anchor the real Stack output. Set fields normally initialized
            # by WhiteDrag first because this path calls the overridden methods.
            self.space=space
            self._exact_jacobian=None;self._exact_center=None;self._exact_params=None
            self._occurrence_matrices=tuple(np.asarray(value,dtype=float) for value in matrices)
            self._directions=tuple(directions)
            self._edited_occurrences=tuple(bool(value) for value in edited_occurrences)
            if not (len(self._occurrence_matrices)==len(self._directions)==len(self._edited_occurrences)):
                raise ValueError('Stack occurrence data must have matching lengths')
            if not any(self._edited_occurrences):
                raise ValueError('The edited Grade has no enabled Stack occurrence')
            # linear.decompose is an LRU-cached math helper whose matrix argument
            # must be immutable/hashable; RNA matrices already are tuples, while
            # test and numeric clients may naturally pass a NumPy array.
            authoritative=np.asarray(authoritative_matrix,dtype=float)
            if authoritative.shape!=(3,3) or not np.isfinite(authoritative).all():
                raise ValueError('Authoritative Stage-1 matrix must be finite and 3 by 3')
            self._authoritative_matrix=tuple(tuple(float(x) for x in row) for row in authoritative)
            self._native_left=np.asarray(native_left,dtype=float)
            self._native_right=np.asarray(native_right,dtype=float)
            self._to_native=np.asarray(to_native,dtype=float)
            self._from_native=np.asarray(from_native,dtype=float)
            self._baseline_total=compose_stage1(self._occurrence_matrices,self._directions)
            # Keep the occurrence shell in production float32 precision. Unedited
            # matrices, including their inverse endpoints, never change in a drag.
            self._matrix_template=tuple(np.asarray(value,dtype='f4').copy()
                                        for value in self._occurrence_matrices)
            # Stack computes inverse endpoints from its float64 native matrices,
            # then StackKernel stores those inverses as float32. Reversing that
            # order changes ill-conditioned inverse responses during a drag.
            self._inverse_template=tuple(np.linalg.inv(value).astype('f4') if direction=='INVERSE'
                                         else None for value,direction in zip(self._occurrence_matrices,self._directions))
            self._stack_model=stack_model
            self._prepared=prepared
            self._input_rgb=None if input_rgb is None else np.asarray(input_rgb,dtype=float).reshape(-1,3)
            sampled_rgb,sampled_alpha=sparse_samples(rgb,alpha)
            has_scope_sample=np.any(np.isfinite(sampled_rgb).all(axis=-1)&
                                    np.isfinite(sampled_alpha)&(sampled_alpha>0))
            self._exact=(self._prepared is not None and self._input_rgb is not None and
                         len(self._input_rgb)==len(np.asarray(rgb).reshape(-1,3)) and
                         has_scope_sample)
            # A matrix-only approximation is not a valid model of a Stack once a
            # nonlinear stage (or an iterative inverse) follows Stage 1. Keep the
            # captured scope visible, but do not steer or predict a drag unless the
            # prepared production evaluator and its matching input snapshot exist.
            self.interactive=self._exact
            self._endpoint_only=(self._exact and len(self._directions)>0 and
                self._directions[-1]=='INVERSE' and
                self._edited_occurrences==tuple(i==len(self._directions)-1
                    for i in range(len(self._directions))))
            self._selected_eval=False
            self._output_cache={}
            if self._exact:
                self._evaluation_native_input=(self._input_rgb@self._to_native.T).astype('f4')
                self._baseline_params=(float(params.source_x),float(params.source_y))
                if self._endpoint_only:
                    raw=self._candidate_execution(params,output_matrix=np.eye(3)).apply_native(
                        self._evaluation_native_input)
                    self._endpoint_core=np.asarray(raw,dtype='f4')
                else:
                    self._endpoint_core=None
                base=self._evaluate_model(params)
                self._baseline_model=base
                self._output_cache[self._baseline_params]=base
            else:
                self._evaluation_native_input=None
                self._baseline_params=(float(params.source_x),float(params.source_y))
                self._baseline_model=None;self._endpoint_core=None
            super().__init__(rgb,alpha,params,space,scope_rgb=rgb)
            self._captured_center=self.initial_center.copy()
            self._exact_jacobian=None;self._exact_center=None;self._exact_params=None
            if self._exact:
                # The initial confidence fit used all captured points. Thereafter
                # only its fixed inliers enter the center or ghost cloud, so an
                # inverse candidate need not solve the excluded pixels.
                indices=self.selection_indices
                self._evaluation_native_input=self._evaluation_native_input[indices].copy()
                self._baseline_model=self._baseline_model[indices].copy()
                if self._endpoint_core is not None:
                    self._endpoint_core=self._endpoint_core[indices].copy()
                self._output_cache={self._baseline_params:self._baseline_model}
                self._selected_eval=True
                self._exact_params=self._baseline_params
                self._exact_center=self._scope_center(self._baseline_model)
                self._exact_jacobian=self._measure_exact_jacobian(params)
            else:
                # `rgb` is the captured installed-Stack output. It is the only
                # trustworthy anchor available when candidate evaluation failed;
                # applying the edited Grade matrix to it would double-apply Stage
                # 1 and can point an inverse gesture in the wrong direction.
                self._captured_center=self.weights@project(self.rgb,self.space)
                self.initial_center=self._captured_center.copy()

        def composed_matrix(self,params):
            """Current total Stage-1 response, including every shared-grade use."""
            edited=cm.edited_matrix(self._authoritative_matrix,self.space,
                {'source_x':float(params.source_x),'source_y':float(params.source_y)})
            candidate=self._native_left@np.asarray(edited,dtype=float)@self._native_right
            matrices=tuple(candidate if target else matrix
                           for matrix,target in zip(self._occurrence_matrices,self._edited_occurrences))
            return compose_stage1(matrices,self._directions)

        def _candidate_matrices(self,params):
            """Return each occurrence matrix after applying this source-white edit."""
            edited=cm.edited_matrix(self._authoritative_matrix,self.space,
                {'source_x':float(params.source_x),'source_y':float(params.source_y)})
            candidate=self._native_left@np.asarray(edited,dtype=float)@self._native_right
            return tuple(candidate if target else matrix
                         for matrix,target in zip(self._occurrence_matrices,self._edited_occurrences))

        def _candidate_execution(self,params,output_matrix=None):
            """Clone only the prepared Stack's matrix shell; never mutate its owner."""
            from copy import copy
            if self._prepared is None:
                raise RuntimeError('The Stack evaluator is unavailable')
            edited=cm.edited_matrix(self._authoritative_matrix,self.space,
                {'source_x':float(params.source_x),'source_y':float(params.source_y)})
            candidate_native=self._native_left@np.asarray(edited,dtype=float)@self._native_right
            candidate=np.asarray(candidate_native,dtype='f4')
            candidate_inverse=(np.linalg.inv(candidate_native).astype('f4')
                               if any(selected and direction=='INVERSE' for selected,direction
                                      in zip(self._edited_occurrences,self._directions)) else None)
            matrices=tuple(candidate if selected else value for selected,value
                           in zip(self._edited_occurrences,self._matrix_template))
            inverses=tuple(candidate_inverse if selected and direction=='INVERSE' else value
                           for selected,direction,value in zip(self._edited_occurrences,
                                                               self._directions,self._inverse_template))
            execution=copy(self._prepared)
            execution.matrices=matrices
            execution.inverses=inverses
            execution.input_matrix=(matrices[0] if self._directions and
                                    self._directions[0]=='FORWARD' else np.eye(3,dtype='f4'))
            execution.output_matrix=(inverses[-1] if self._directions and
                                     self._directions[-1]=='INVERSE' else np.eye(3,dtype='f4'))
            if output_matrix is not None:execution.output_matrix=np.asarray(output_matrix,dtype='f4')
            # The UI always uses CPU evaluation. Do not share a GPU dispatch object
            # with a bake or another editor.
            execution.program=None
            return execution

        def _evaluate_model(self,params):
            if self._endpoint_only:
                matrix=np.asarray(self._candidate_matrices(params)[-1],dtype=float)
                inverse=np.linalg.inv(matrix).astype('f4')
                # Use the production float32 matrix kernel (including its chosen
                # evaluation order) for parity with StackKernel's output endpoint.
                native=self._prepared.matrix(np.asarray(self._endpoint_core,dtype='f4'),inverse)
            else:
                execution=self._candidate_execution(params)
                native=execution.apply_native(self._evaluation_native_input)
            return np.asarray(native,dtype=float)@self._from_native.T

        def _model_output(self,params):
            if not self._exact:return None
            key=(float(params.source_x),float(params.source_y))
            value=self._output_cache.get(key)
            if value is None:
                value=self._evaluate_model(params)
                self._output_cache[key]=value
                if len(self._output_cache)>64:self._output_cache.pop(next(iter(self._output_cache)))
            return value

        def _scope_center(self,output):
            predicted=self.rgb+np.asarray(output)-self._baseline_model
            return self.weights@project(predicted,self.space)

        def _measure_exact_jacobian(self,params):
            point=np.array((params.source_x,params.source_y),dtype=float);columns=[]
            for axis in range(2):
                delta=np.eye(2)[axis]*1e-4
                low=cm.source_component(point-delta,axis)
                high=cm.source_component(point+delta,axis)
                span=high[axis]-low[axis]
                if span<=1e-12:
                    columns.append(np.zeros(2));continue
                low_params=replace(params,source_x=low[0],source_y=low[1])
                high_params=replace(params,source_x=high[0],source_y=high[1])
                columns.append((self._scope_center(self._model_output(high_params))-
                                self._scope_center(self._model_output(low_params)))/span)
            return np.stack(columns,axis=-1)

        def _exact_position(self,params):
            output=self._model_output(params);center=self._scope_center(output)
            point=np.array((float(params.source_x),float(params.source_y)))
            if self._exact_params is not None and self._exact_center is not None:
                delta=point-np.asarray(self._exact_params)
                square=float(delta@delta)
                if square>1e-16 and self._exact_jacobian is not None:
                    residual=center-self._exact_center-self._exact_jacobian@delta
                    self._exact_jacobian+=np.outer(residual,delta)/square
            self._exact_params=tuple(point)
            self._exact_center=np.asarray(center,dtype=float)
            return self._exact_center.copy()

        def matrix_jacobian(self,params):
            """Finite-difference d(composed matrix)/d(source_x, source_y)."""
            result=[]
            point=np.array((params.source_x,params.source_y),dtype=float)
            for axis in range(2):
                minus=cm.source_component(point-1e-5*np.eye(2)[axis],axis)
                plus=cm.source_component(point+1e-5*np.eye(2)[axis],axis)
                span=plus[axis]-minus[axis]
                if span<=1e-14:
                    result.append(np.zeros((3,3)));continue
                low=replace(params,source_x=minus[0],source_y=minus[1])
                high=replace(params,source_x=plus[0],source_y=plus[1])
                result.append((self.composed_matrix(high)-self.composed_matrix(low))/span)
            return np.stack(result,axis=-1)

        def preview_rgb(self,params):
            """Predicted whole-Stack output samples in GradeSpace RGB coordinates."""
            try:
                if self._exact:
                    output=self._model_output(params)
                    if not self._selected_eval:
                        return self.rgb+output[self.selection_indices]-self._baseline_model[self.selection_indices]
                    return self.rgb+output-self._baseline_model
                return self.rgb.copy()
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                return self.rgb.copy()

        def preview_coordinates(self,params):
            """Vectorscope coordinates for the transient whole-Stack ghost cloud."""
            return project(self.preview_rgb(params),self.space)

        def center(self,params=None):
            """Return captured actual center, or a predicted center for parameters."""
            if params is None or not self._exact:return self._captured_center.copy()
            return self._exact_position(params)

        def position(self,params):
            # Steering and its short backtracking line search use a local model
            # updated from exact whole-Stack samples after every accepted move.
            # This avoids running the iterative inverse evaluator several times
            # for one pointer event.
            if not self._exact:
                # WhiteDrag.__init__ asks for an initial position after it has
                # populated the selected samples and weights, but before this
                # subclass has stored _captured_center.
                return self.weights@project(self.rgb,self.space)
            if self._exact_center is not None and self._exact_params is not None:
                delta=np.array((params.source_x,params.source_y))-np.asarray(self._exact_params)
                if self._exact_jacobian is not None:return self._exact_center+self._exact_jacobian@delta
                return self._exact_center.copy()
            return self.weights@project(self.preview_rgb(params),self.space)

        def jacobian(self,params,rgb=None,weights=None):
            """Use the exact local Stack derivative when candidate evaluation exists."""
            if not self._exact:return np.zeros((2,2),dtype=float)
            if self._exact and self._exact_jacobian is not None:
                point=(float(params.source_x),float(params.source_y))
                if point==self._exact_params:return self._exact_jacobian.copy()
            point=np.array((params.source_x,params.source_y),dtype=float);columns=[]
            for axis in range(2):
                minus=cm.source_component(point-1e-5*np.eye(2)[axis],axis)
                plus=cm.source_component(point+1e-5*np.eye(2)[axis],axis)
                span=plus[axis]-minus[axis]
                if span<=1e-14:
                    columns.append(np.zeros(2));continue
                low=replace(params,source_x=minus[0],source_y=minus[1])
                high=replace(params,source_x=plus[0],source_y=plus[1])
                columns.append((self.position(high)-self.position(low))/span)
            return np.stack(columns,axis=-1)

        def move_pixels(self,params,pointer_delta,xy_per_pixel):
            """Disable Stack edits when no exact production evaluator is available."""
            if not self._exact:
                return dict(source_x=float(params.source_x),source_y=float(params.source_y))
            return super().move_pixels(params,pointer_delta,xy_per_pixel)


    def from_node(node,space,params):
        scopes = _application.component('scopes.sampling')
        spaces = _application.component('graph.spaces'); nodes = _application.component('graph.nodes')
        editing = _application.component('graph.editing')
        if editing.is_target(node):
            return _from_stack_target(node,space,params,scopes,spaces,nodes)
        try:
            rgba,chain,_,_=scopes.source_samples(node,space,size=SAMPLE_GRID)
            rgb=scopes.apply_chain(rgba[...,:3],chain,space);alpha=rgba[...,3]
            rgb=rgb@spaces.adapters(nodes.working_space())[0].T
        except (ValueError,RuntimeError,ReferenceError,TypeError,OSError):
            rgb=np.full((1,3),.18);alpha=np.ones(1)
        return WhiteDrag(rgb,alpha,params,nodes.grading_space(node))


    def _from_stack_target(target,space,params,scopes,spaces,nodes):
        """Capture an installed Stack scope and its Stage-1 response model."""
        cube_nodes = _application.component('graph.lut'); stacks = _application.component('graph.stacks')
        from ...perceptual_grade_core.model import lut as model_lut
        stack=target.stack;working=nodes.working_space()
        items=[item for item in stack.pg_stack_entries
               if item.enabled and item.grade is not None]
        if not items:
            raise ValueError('The edited Stack has no enabled Grade entries')
        stack_model=stacks.model(stack)
        grade=target.pg_grade
        occurrences=tuple(item.grade==grade for item in items)
        if len(occurrences)!=len(stack_model.grades):
            raise ValueError('Stack entries changed while preparing the vectorscope drag')
        if not any(occurrences):
            raise ValueError('The edited Grade has no enabled Stack occurrence')

        # The Stack scope is the installed whole-Stack LUT. Match its sparse grid
        # exactly, then express its output in the vectorscope's canonical RGB basis.
        preview_available=True
        try:
            rgba,chain,_,_=scopes.source_samples(stack,working,size=SAMPLE_GRID)
            incoming=scopes.apply_chain(rgba[...,:3],chain,working)
            signal=cube_nodes.installed_signal(stack)
            output=cube_nodes.evaluate_signal(incoming,signal)
            rgb=output@spaces.adapters(working)[0].T
            alpha=rgba[...,3]
            if not np.isfinite(rgb).all():raise ValueError('Stack scope samples are nonfinite')
        except (ValueError,RuntimeError,ReferenceError,TypeError,OSError):
            # Retain the existing robust fallback for a temporarily unavailable
            # scope. The diagram can suppress the transient ghost in this case.
            rgb=np.full((*SAMPLE_GRID[::-1],3),.18);alpha=np.ones(SAMPLE_GRID[::-1])
            preview_available=False

        forwards=tuple(item.forward for item in stack_model.grades)
        matrices=tuple(np.asarray(forward.matrix_native,dtype=float) for forward in forwards)
        directions=tuple(stack_model.directions)
        selected=next(i for i,value in enumerate(occurrences) if value)
        forward=forwards[selected]
        boundary=np.asarray(forward.boundary.matrix,dtype=float)
        native_left=boundary@(np.asarray(cm.CAT16,dtype=float)/forward.white_cones[:,None])
        native_right=forward.cones_to_xyz@np.linalg.inv(boundary)
        to_native,from_native=model_lut.basis(nodes.grading_space(target))
        # Reuse the active baker's prepared immutable nonlinear execution when it
        # has the same Stack core. Endpoint matrices may differ because they are
        # excluded from that core; StackWhiteDrag updates a private shallow copy.
        prepared=None
        try:
            state=cube_nodes._states.get(stack.as_pointer())
            baker=state.get('baker') if state is not None else None
            if (baker is not None and getattr(baker,'stack',False) and
                    getattr(baker,'execution',None) is not None and
                    getattr(getattr(baker,'model',None),'bake_key',None)==stack_model.bake_key):
                prepared=baker.execution
            if prepared is None:prepared=stack_model.prepare()
        except (ValueError,RuntimeError,ArithmeticError,ReferenceError,np.linalg.LinAlgError):
            prepared=None
        input_rgb=incoming@spaces.adapters(working)[0].T if preview_available else None
        drag=StackWhiteDrag(rgb,alpha,params,nodes.grading_space(target),matrices,directions,
            occurrences,nodes.grades.matrix(target),native_left,native_right,to_native,from_native,
            stack_model,input_rgb,prepared)
        drag.preview_available=preview_available
        return drag

    def _set_export_BOUNDARY_BLEND_WIDTH(value):
        nonlocal BOUNDARY_BLEND_WIDTH
        BOUNDARY_BLEND_WIDTH = value

    def _set_export_BOUNDARY_TOLERANCE(value):
        nonlocal BOUNDARY_TOLERANCE
        BOUNDARY_TOLERANCE = value

    def _set_export_DRAG_GAIN(value):
        nonlocal DRAG_GAIN
        DRAG_GAIN = value

    def _set_export_SAMPLE_GRID(value):
        nonlocal SAMPLE_GRID
        SAMPLE_GRID = value

    def _set_export_SAMPLE_LIMIT(value):
        nonlocal SAMPLE_LIMIT
        SAMPLE_LIMIT = value

    def _set_export_SCOPE_MATRIX(value):
        nonlocal SCOPE_MATRIX
        SCOPE_MATRIX = value

    def _set_export_WHITE_NORMALS(value):
        nonlocal WHITE_NORMALS
        WHITE_NORMALS = value

    def _set_export_smooth01(value):
        nonlocal smooth01
        smooth01 = value

    def _set_export_boundary_move(value):
        nonlocal boundary_move
        boundary_move = value

    def _set_export_sparse_samples(value):
        nonlocal sparse_samples
        sparse_samples = value

    def _set_export_confidence_membership(value):
        nonlocal confidence_membership
        confidence_membership = value

    def _set_export_WhiteDrag(value):
        nonlocal WhiteDrag
        WhiteDrag = value

    def _set_export_compose_stage1(value):
        nonlocal compose_stage1
        compose_stage1 = value

    def _set_export_StackWhiteDrag(value):
        nonlocal StackWhiteDrag
        StackWhiteDrag = value

    def _set_export_from_node(value):
        nonlocal from_node
        from_node = value

    def _set_export__from_stack_target(value):
        nonlocal _from_stack_target
        _from_stack_target = value

    return Component(readers={
        "__name__": lambda: __name__,
        'BOUNDARY_BLEND_WIDTH': lambda: BOUNDARY_BLEND_WIDTH,
        'BOUNDARY_TOLERANCE': lambda: BOUNDARY_TOLERANCE,
        'DRAG_GAIN': lambda: DRAG_GAIN,
        'SAMPLE_GRID': lambda: SAMPLE_GRID,
        'SAMPLE_LIMIT': lambda: SAMPLE_LIMIT,
        'SCOPE_MATRIX': lambda: SCOPE_MATRIX,
        'StackWhiteDrag': lambda: StackWhiteDrag,
        'WHITE_NORMALS': lambda: WHITE_NORMALS,
        'WhiteDrag': lambda: WhiteDrag,
        '_from_stack_target': lambda: _from_stack_target,
        'boundary_move': lambda: boundary_move,
        'cm': lambda: cm,
        'compose_stage1': lambda: compose_stage1,
        'confidence_membership': lambda: confidence_membership,
        'from_node': lambda: from_node,
        'np': lambda: np,
        'project': lambda: project,
        'replace': lambda: replace,
        'rs': lambda: rs,
        'smooth01': lambda: smooth01,
        'sparse_samples': lambda: sparse_samples,
    }, writers={
        'BOUNDARY_BLEND_WIDTH': _set_export_BOUNDARY_BLEND_WIDTH,
        'BOUNDARY_TOLERANCE': _set_export_BOUNDARY_TOLERANCE,
        'DRAG_GAIN': _set_export_DRAG_GAIN,
        'SAMPLE_GRID': _set_export_SAMPLE_GRID,
        'SAMPLE_LIMIT': _set_export_SAMPLE_LIMIT,
        'SCOPE_MATRIX': _set_export_SCOPE_MATRIX,
        'WHITE_NORMALS': _set_export_WHITE_NORMALS,
    'smooth01': _set_export_smooth01, 'boundary_move': _set_export_boundary_move, 'sparse_samples': _set_export_sparse_samples, 'confidence_membership': _set_export_confidence_membership, 'WhiteDrag': _set_export_WhiteDrag, 'compose_stage1': _set_export_compose_stage1, 'StackWhiteDrag': _set_export_StackWhiteDrag, 'from_node': _set_export_from_node, '_from_stack_target': _set_export__from_stack_target})
