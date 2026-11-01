"""Ordinary typed machine bindings for one product Session."""
from ..invariant import Machine
from ..blender_runtime.machines import create_execution_machines, create_ownership_machine, create_edge_effect_machine, create_native_evaluation_machine
from ..perceptual_grade_core.graph.policies import channel, lifecycle, exact, display, output, bake_image
from ..perceptual_grade_core.graph.coordinators import bake as bake_workflow, scope as scope_workflow, publication, exact as exact_workflow, scope_input, scope_demand, bake_queue, recovery
from importlib import import_module


def _machine(module, *, initializers=('new',), queries=()):
    parent, name = module.__name__.rsplit('.', 1)
    return Machine(module, contracts=import_module(parent + '._' + name + '_contracts'),
                   initializers=initializers, queries=queries)


def create_machines():
    execution = create_execution_machines()
    return {
        'attempt': execution.attempt,
        'gpu': execution.gpu,
        'retry': execution.retry,
        'startup': execution.startup,
        'ownership': create_ownership_machine(),
        'edge_effect': create_edge_effect_machine(),
        'native_evaluation': create_native_evaluation_machine(),
        'channel': _machine(channel, initializers=('initial',), queries=(
            'replacement_decision', 'linear_scope_request',
            'scope_job_admissible', 'scope_ack_admissible',
            'scope_cycle', 'final_scope_resolution', 'scope_next')),
        'lifecycle': _machine(lifecycle, queries=('lifecycle_admits',)),
        'exact': _machine(exact),
        'display': _machine(display, queries=(
            'scope_display_statistics_match', 'scope_display_current')),
        'output': _machine(output, initializers=('establish',), queries=(
            'complete', 'output_complete', 'packing_clean', 'slot_for')),
        'bake_image': _machine(bake_image, queries=('settled', 'recovery_required')),
        'bake_coordinator': _machine(bake_workflow),
        'scope_coordinator': _machine(scope_workflow),
        'scope_input_coordinator': _machine(scope_input),
        'scope_demand_coordinator': _machine(scope_demand),
        'recovery_coordinator': _machine(recovery),
        'bake_queue_coordinator': _machine(bake_queue, queries=('target_busy', 'lane_busy', 'destination_waits')),
        'publication_coordinator': _machine(publication),
        'exact_coordinator': _machine(exact_workflow),
    }
