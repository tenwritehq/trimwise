# Current native CPU verification

Both framework-local examples have their own isolated environment and native pipeline.
The current model-free shared suite passed **40 tests**, with one inapplicable
LangChain parameter skipped for the LlamaIndex-only native-field contract. The
adapter suites passed 18 LangChain and 20 LlamaIndex tests. Ruff check/format,
strict mypy (Python 3.13), and SDK/private-helper sdist and wheel builds passed.
Built archives were checked for virtual environments, caches, and local results.
An independent review rechecked the repaired native-object boundary and packaging.

`cpu-native-smoke.json` records one public incident question through this framework,
with full content explicitly included for the synthetic fixture. Both native apps
also completed the same comparison in the browser after the final pipeline repair:
904 rendered context tokens became 278 (69.2% smaller), and both answers correctly
reported recovery at 14:32 UTC. Full native prompt sizes matched runtime usage.
Ollama reported **zero model VRAM**, with `num_gpu=0` in preload and generation.
This is a small integration smoke check, not a quality or speed benchmark. The
complete five-question smoke command remains available in the parent README.

`native-cpu-desktop.png` shows the current framework-specific UI with square or
2px corners. Shared responsive controls were checked at 320, 375, 414, and 768px,
including the assessment selector and wrapping action button. The two apps were
checked independently for native indexing and answer generation. Core Trimwise
source and runtime dependencies were unchanged. These verification commands do not publish packages.
