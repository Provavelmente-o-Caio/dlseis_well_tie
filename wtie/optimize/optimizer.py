"""Utils function using the tool Ax to perform global optimization"""

import torch

from ax.service.ax_client import AxClient

# Support multiple ax-platform layouts: prefer registry, fall back to
# factory. Raise clear ImportError if neither is importable.
try:
    from ax.modelbridge.registry import Models
    from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
except ModuleNotFoundError:
    try:
        from ax.modelbridge.factory import Models
        from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
    except ModuleNotFoundError:
        try:
            from ax.adapter.registry import Generators as Models
            from ax.generation_strategy.generation_strategy import (
                GenerationStep,
                GenerationStrategy,
            )
        except ModuleNotFoundError:
            raise ImportError(
                "Could not import ax.modelbridge registry or factory, "
                "nor ax.adapter.registry/ax.generation_strategy. "
                "Install a compatible ax-platform version or adjust imports."
            )



def create_ax_client(num_iters: int,
                     random_ratio: float=0.6,
                     verbose: bool=False,
                     **kwargs) -> AxClient:
    """Default client, first sobol, then bayes. """
    n_sobol = int(random_ratio*num_iters)
    n_bayes = num_iters - n_sobol

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Normalize Models enum members across ax versions
    def _resolve_model(*names):
        for n in names:
            if hasattr(Models, n):
                return getattr(Models, n)
        return None

    SOBOL_MODEL = _resolve_model("SOBOL", "Sobol")
    BOTORCH_MODEL = _resolve_model("BOTORCH", "BOTORCH_MODULAR", "BoTorch")
    if BOTORCH_MODEL is None:
        raise ImportError(
            "Could not find a BoTorch generator in ax Models enum (tried 'BOTORCH', 'BOTORCH_MODULAR', 'BoTorch'). "
            "Please install a compatible ax-platform or update the code to use available generators."
        )

    # Only pass kwargs that adapters/generators expect. `torch_device` is
    # accepted by `TorchAdapter`; `torch_dtype` is not accepted by the
    # current adapter/generator callables and caused a ValueError, so omit it.
    ax_gen_startegy = GenerationStrategy(
        [
            GenerationStep(SOBOL_MODEL, num_trials=n_sobol),
            GenerationStep(
                BOTORCH_MODEL,
                num_trials=n_bayes,
                max_parallelism=8,
                model_kwargs={"torch_device": device},
            ),
        ]
    )

    return AxClient(generation_strategy=ax_gen_startegy,
                    verbose_logging=verbose, **kwargs)