"""Named, immutable identities used by scope capture and scheduling.

The identity is deliberately value-only.  It may cross the helper boundary
through the explicit execution codec, but it never contains Blender objects,
arrays or callbacks.  The projection methods preserve the old semantic
partitions while making their membership explicit.
"""
from dataclasses import dataclass, fields, replace


def validate_identity(value):
    """Reject mutable/native fields before retaining or transporting an identity."""
    from ..model import tone, color, linear
    records = (ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity,
               tone.Parameters, color.Parameters, linear.Parameters,
               linear.GradeSpace)
    active = set()

    def visit(item):
        kind = type(item)
        if item is None or kind in (bool, int, float, str):
            return
        token = id(item)
        if token in active:
            raise ValueError('Scope identities cannot contain cycles')
        active.add(token)
        try:
            if kind is tuple:
                for entry in item:
                    visit(entry)
            elif kind in records:
                for field in fields(item):
                    visit(getattr(item, field.name))
            else:
                raise TypeError('Unsupported scope identity field: ' + kind.__name__)
        finally:
            active.remove(token)

    if type(value) not in (ScopeIdentity, ActiveScopeIdentity, LinearScopeIdentity):
        raise TypeError('Expected a named scope identity')
    visit(value)


@dataclass(frozen=True, slots=True)
class ActiveScopeIdentity:
    """Presentation demand that changes raster publication semantics."""

    schema: int
    requested: tuple
    views: tuple
    enabled: bool


@dataclass(frozen=True, slots=True)
class LinearScopeIdentity:
    """Stage 1 request, independent of tone/chromatic control changes."""

    source: object
    reference: object
    matrix: object


@dataclass(frozen=True, slots=True)
class ScopeIdentity:
    """Complete scope identity with named source and presentation fields."""

    version: object
    reference_version: object
    comparison: object
    matrix: object
    tone: object
    request_params: object
    stack_signature: object
    generation: int
    stage: int
    presentation: ActiveScopeIdentity | None = None

    def with_presentation(self, presentation):
        if presentation is not None and type(presentation) is not ActiveScopeIdentity:
            raise TypeError('Scope presentation must be an ActiveScopeIdentity')
        return replace(self, presentation=presentation)

    def analysis(self):
        """Identity used by reusable numerical analysis, excluding raster UI."""
        return replace(self, presentation=None)

    def source_projection(self, stack=False):
        """Raw source identity used by capture/cache ownership."""
        return (self.version, self.reference_version, self.comparison,
                self.stack_signature if stack else None)

    def scheduler_projection(self, evaluation):
        """Semantic scheduler identity, intentionally excluding stack signature."""
        return ((self.version, self.reference_version, self.comparison,
                 self.matrix, self.tone, self.request_params),
                (self.generation, self.stage, self.presentation), evaluation)

    def capture_projection(self, owner, context, scene_space, grading_space):
        """Identity for a captured numerical source envelope."""
        return (owner, context, scene_space, grading_space, self.version,
                self.matrix, self.tone, self.generation,
                self.stack_signature)

