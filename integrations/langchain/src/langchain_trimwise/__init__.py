"""Expose the shared-budget Trimwise Runnable for LangChain documents."""

from langchain_trimwise.runnable import TrimwiseInput, create_trimwise_runnable

__all__ = ["TrimwiseInput", "create_trimwise_runnable"]
