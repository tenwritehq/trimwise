"""Trim a real LlamaIndex retrieval result without downloading or calling models."""

from llama_index.core import SummaryIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.postprocessor.trimwise import TrimwiseNodePostprocessor


def main() -> None:
    """Retrieve a tiny fixture, apply one budget, and inspect retained source spans."""
    nodes = [
        TextNode(
            id_="garden-1",
            text="Gardens have flowers and trees. " * 10,
            metadata={"source": "garden.pdf", "page": 1},
        ),
        TextNode(
            id_="manual-3",
            text="The retry deadline is thirty seconds.",
            metadata={"source": "manual.pdf", "page": 3},
        ),
    ]
    question = "What is the retry deadline?"
    # SummaryIndex's default retriever returns every node; no embedding model is needed.
    retrieved = SummaryIndex(nodes).as_retriever().retrieve(question)
    processor = TrimwiseNodePostprocessor(limit=12, unit="words")
    retained = processor.postprocess_nodes(retrieved, query_str=question)
    assert sum(len(item.node.get_content(MetadataMode.NONE).split()) for item in retained) <= 12
    assert any("thirty seconds" in item.node.get_content(MetadataMode.NONE) for item in retained)
    assert all("trimwise" not in node.metadata for node in nodes)
    for item in retained:
        print(f"[{item.node.metadata['source']}, page {item.node.metadata['page']}]")
        print(item.node.get_content(MetadataMode.NONE))
        print("Provenance:", item.node.metadata["trimwise"])


if __name__ == "__main__":
    main()
