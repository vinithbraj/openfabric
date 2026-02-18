from .adapters.mock_llm import MockLLM
from .adapters.mock_retrieval import MockRetrieval


ADAPTER_REGISTRY = {
    "mock_llm": MockLLM,
    "mock_retrieval": MockRetrieval
}
