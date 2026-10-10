"""Common trial-boundary FIFO adapter. Central policy belongs to coordinator."""
from contextlib import contextmanager
from .common import HERE, TRAIN_STOP
from experiments.resource_coordination.scheduler import lease as common_lease, configure_torch, alive

@contextmanager
def lease(purpose, owner_directory=HERE):
    with common_lease(HERE.name, purpose, deadline=TRAIN_STOP, owner_directory=owner_directory) as resource:
        yield resource
