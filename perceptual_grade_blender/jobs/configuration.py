"""Perceptual Grade adapters for the privately bundled execution service."""
from ...blender_runtime.execution.contracts import RuntimeProfile
from ...perceptual_grade_core.jobs.values import seal

storage_name = 'perceptual_grade'
profile = RuntimeProfile(lanes=('linear', 'bulk', 'gpu', 'scope_gpu', 'capture', 'bake_0', 'bake_1'),
                         gpu_probes=(('gpu_probe', 'gpu'), ('scope_probe', 'scope_gpu')),
                         uncached_operations=('scope_register', 'scope_capture', 'scope_image_read', 'compositor_bake', 'shader_bake'), dispatcher='perceptual_grade_blender.jobs.operations')
GPU_ROLES = tuple(role for _, role in profile.gpu_probes)
