"""Package-specific exceptions."""


class DistilQwenError(RuntimeError):
    """Base error raised by distil-qwen."""


class ModelContractError(DistilQwenError):
    """Raised when a model does not expose the expected Qwen3-ASR structure."""


class OptionalDependencyError(DistilQwenError, ImportError):
    """Raised when an optional feature dependency is unavailable."""


class BackendConfigurationError(DistilQwenError, ValueError):
    """Raised when an inference backend is configured inconsistently."""


class BackendFeatureError(DistilQwenError, NotImplementedError):
    """Raised when an upstream backend cannot provide a requested feature."""


class BackendRequestError(DistilQwenError):
    """Raised when an inference server rejects or cannot complete a request."""
