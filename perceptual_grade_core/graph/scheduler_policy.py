"""Scheduling vocabulary and pure query adapters.

State transitions are implemented in the policy modules and committed by
explicit App-owned Stores.
"""
from .policies import channel as _channel, lifecycle as _lifecycle
from .policies import exact as _exact, display as _display

SCHEMA = 10
KINDS = ('begin', 'request', 'release', 'signal', 'ready', 'publish',
         'start', 'complete', 'replace', 'invalidate', 'shutdown', 'recover',
         'retry', 'fail', 'effect_begin', 'effect_commit', 'effect_fail')
LIFECYCLE_KINDS = ('activate','quiesce','save_begin','save_ready','save_done',
                   'recover','recovered','fail','crash','disable')
EXACT_KINDS = ('begin', 'changed', 'validated', 'native_begin', 'native_result',
               'native_stopped', 'cleanup_failed', 'settled', 'next_frame', 'cancel', 'fail')
EXACT_IDLE, EXACT_PREPARING, EXACT_READY, EXACT_NATIVE, EXACT_SETTLING, EXACT_COMPLETE, EXACT_FAILED, EXACT_CANCELLED = range(8)
SCOPE_DISPLAY_KINDS = ('request', 'visual', 'statistics', 'invalidate',
                       'release', 'cancel', 'failure', 'retry', 'reset', 'geometry')

State, Event, Effects = _channel.State, _channel.Event, _channel.Effects
ScopeCycle = _channel.ScopeCycle
LifecycleState, LifecycleEvent = _lifecycle.LifecycleState, _lifecycle.Event
ExactOperationState, ExactEvent, ExactEffects = _exact.ExactOperationState, _exact.ExactEvent, _exact.ExactEffects
ScopeDisplayState, ScopeDisplayEvent, ScopeDisplayEffects = _display.ScopeDisplayState, _display.ScopeDisplayEvent, _display.ScopeDisplayEffects
linear_scope_request = _channel.linear_scope_request
scope_job_admissible = _channel.scope_job_admissible
scope_ack_admissible = _channel.scope_ack_admissible
lifecycle_admits = _lifecycle.lifecycle_admits
scope_display_statistics_match = _display.scope_display_statistics_match


def initial(owner=0, key=None, session=1, scope=False, linear=False):
    return _channel.initial(owner, 0 if key is None else key, session, scope, linear)


def replacement_decision(requested, active, dragging, started, preview, finished,
                         published=False, cancelled=False, scope_demand=False):
    return _channel.replacement_decision(requested, active, dragging, started,
        preview, finished, published, cancelled, scope_demand)


def scope_cycle(wanted, compatible, exact, dragging, retired=False, preview=False,
                control=False, ack=False, pending=False, history=False):
    return _channel.scope_cycle(wanted, compatible, exact, dragging, retired,
        preview, control, ack, pending, history)


def scope_next(eligible, start):
    return _channel.scope_next(tuple(eligible), start)


def final_scope_resolution(resolutions):
    return _channel.final_scope_resolution(tuple(resolutions))


def scope_display_current(state, statistics_required=False):
    return _display.scope_display_current(state, statistics_required)
