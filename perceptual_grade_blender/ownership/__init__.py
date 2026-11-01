"""Native ownership capabilities bound to this exact product Session."""

from ..integration.components import Component

def create(_application):
    from ...blender_runtime.ownership import NativeRef
    from ...blender_runtime.ownership import OwnedField as FrameworkOwnedField, bind_holder

    arena = _application.app.ownership
    retain = arena.retain
    resolve = arena.resolve
    release = arena.release
    borrow = arena.borrow
    generation = arena.generation
    generation_key = arena.generation_key
    observation = arena.observation
    snapshots = arena.snapshots
    detach_all = arena.detach_all
    begin_boundary = arena.begin_boundary
    native_replaced = arena.native_replaced
    defer_pending_replacement = arena.defer_pending_replacement
    recovered = arena.recovered
    retry_recovery = arena.retry_recovery
    fail = arena.fail
    admits = arena.admits
    require_ready = arena.require_ready
    is_current = arena.is_current
    retire = arena.retire
    retire_datablock = arena.retire_datablock


    def Arena():
        return arena.arena()


    class OwnedField(FrameworkOwnedField):
        """A retained Blender class keeps its original Session's context binding."""
        def _context(self, instance):
            bind_holder(instance, arena)
            return super()._context(instance)


    def fresh_session():
        _application.app.require_open()
        if arena.snapshots():
            raise RuntimeError('A fresh ownership lifecycle requires a new App after terminal cleanup')


    def configure_keys(keys):
        if keys is not arena.config.keys:
            raise RuntimeError('Native identity projection must be configured when constructing the App')

    def _set_export_admits(value):
        nonlocal admits
        admits = value

    def _set_export_arena(value):
        nonlocal arena
        arena = value

    def _set_export_begin_boundary(value):
        nonlocal begin_boundary
        begin_boundary = value

    def _set_export_borrow(value):
        nonlocal borrow
        borrow = value

    def _set_export_defer_pending_replacement(value):
        nonlocal defer_pending_replacement
        defer_pending_replacement = value

    def _set_export_detach_all(value):
        nonlocal detach_all
        detach_all = value

    def _set_export_fail(value):
        nonlocal fail
        fail = value

    def _set_export_generation(value):
        nonlocal generation
        generation = value

    def _set_export_generation_key(value):
        nonlocal generation_key
        generation_key = value

    def _set_export_is_current(value):
        nonlocal is_current
        is_current = value

    def _set_export_native_replaced(value):
        nonlocal native_replaced
        native_replaced = value

    def _set_export_recovered(value):
        nonlocal recovered
        recovered = value

    def _set_export_release(value):
        nonlocal release
        release = value

    def _set_export_require_ready(value):
        nonlocal require_ready
        require_ready = value

    def _set_export_resolve(value):
        nonlocal resolve
        resolve = value

    def _set_export_retain(value):
        nonlocal retain
        retain = value

    def _set_export_retire(value):
        nonlocal retire
        retire = value

    def _set_export_retire_datablock(value):
        nonlocal retire_datablock
        retire_datablock = value

    def _set_export_snapshots(value):
        nonlocal snapshots
        snapshots = value

    def _set_export_Arena(value):
        nonlocal Arena
        Arena = value

    def _set_export_OwnedField(value):
        nonlocal OwnedField
        OwnedField = value

    def _set_export_fresh_session(value):
        nonlocal fresh_session
        fresh_session = value

    def _set_export_configure_keys(value):
        nonlocal configure_keys
        configure_keys = value

    return Component(readers={
        "__name__": lambda: __name__,
        'Arena': lambda: Arena,
        'FrameworkOwnedField': lambda: FrameworkOwnedField,
        'NativeRef': lambda: NativeRef,
        'OwnedField': lambda: OwnedField,
        'admits': lambda: admits,
        'arena': lambda: arena,
        'begin_boundary': lambda: begin_boundary,
        'bind_holder': lambda: bind_holder,
        'borrow': lambda: borrow,
        'configure_keys': lambda: configure_keys,
        'defer_pending_replacement': lambda: defer_pending_replacement,
        'detach_all': lambda: detach_all,
        'fail': lambda: fail,
        'fresh_session': lambda: fresh_session,
        'generation': lambda: generation,
        'generation_key': lambda: generation_key,
        'observation': lambda: observation,
        'is_current': lambda: is_current,
        'native_replaced': lambda: native_replaced,
        'recovered': lambda: recovered,
        'retry_recovery': lambda: retry_recovery,
        'release': lambda: release,
        'require_ready': lambda: require_ready,
        'resolve': lambda: resolve,
        'retain': lambda: retain,
        'retire': lambda: retire,
        'retire_datablock': lambda: retire_datablock,
        'snapshots': lambda: snapshots,
    }, writers={
        'admits': _set_export_admits,
        'arena': _set_export_arena,
        'begin_boundary': _set_export_begin_boundary,
        'borrow': _set_export_borrow,
        'defer_pending_replacement': _set_export_defer_pending_replacement,
        'detach_all': _set_export_detach_all,
        'fail': _set_export_fail,
        'generation': _set_export_generation,
        'generation_key': _set_export_generation_key,
        'is_current': _set_export_is_current,
        'native_replaced': _set_export_native_replaced,
        'recovered': _set_export_recovered,
        'release': _set_export_release,
        'require_ready': _set_export_require_ready,
        'resolve': _set_export_resolve,
        'retain': _set_export_retain,
        'retire': _set_export_retire,
        'retire_datablock': _set_export_retire_datablock,
        'snapshots': _set_export_snapshots,
    'Arena': _set_export_Arena, 'OwnedField': _set_export_OwnedField, 'fresh_session': _set_export_fresh_session, 'configure_keys': _set_export_configure_keys})
