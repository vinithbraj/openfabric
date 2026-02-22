from .adapters.mock_llm import MockLLM
from .adapters.mock_retrieval import MockRetrieval
from .adapters.http_adapter import HTTPAdapter


ADAPTER_REGISTRY = {
    "mock_llm": MockLLM,
    "mock_retrieval": MockRetrieval,
    "http": HTTPAdapter
}
