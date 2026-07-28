.PHONY: install smoke run qa aggregate paired-stats plot test

install:
	uv sync --python 3.12

smoke:
	CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression --limit 1

run:
	CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runners.run_compression

qa:
	uv run python -m benchmark.runners.run_openai_qa

aggregate:
	uv run python -m benchmark.runners.aggregate --input results/position_controlled_160_results.jsonl --dataset data/position_controlled_160.jsonl --qa-input results/position_controlled_160_qa_gpt_5_4_nano.jsonl --qa-input results/position_controlled_160_qa_gpt_5_4_mini.jsonl --qa-input results/position_controlled_160_qa_gpt_5_6_luna.jsonl --output results/position_controlled_160_summary.csv

paired-stats:
	uv run python scripts/paired_stats.py

plot:
	uv run python -m benchmark.runners.plot --input results/position_controlled_160_summary.csv --output-dir reports

test:
	uv run pytest
