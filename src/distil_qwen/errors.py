"""Package-specific exceptions."""


class DistilQwenError(RuntimeError):
    """Base error raised by distil-qwen."""


class ModelContractError(DistilQwenError):
    """Raised when a model does not expose the expected Qwen3-ASR structure."""


class OptionalDependencyError(DistilQwenError, ImportError):
    """Raised when an optional feature dependency is unavailable."""
