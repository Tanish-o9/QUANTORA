"""tests/test_evaluator.py - Unit tests for Quantora's quality evaluation module."""

import pytest
from evaluator import get_evaluator


def test_evaluator_singleton():
    """Verify singleton instance consistency."""
    e1 = get_evaluator()
    e2 = get_evaluator()
    assert e1 is e2


def test_rouge_identical_text():
    """Identical prediction and reference must achieve 100% ROUGE."""
    evaluator = get_evaluator()
    text = "The quick brown fox jumps over the lazy dog."
    scores = evaluator.compute_rouge(text, text)

    assert scores["rouge1"] == 100.0
    assert scores["rouge2"] == 100.0
    assert scores["rougeL"] == 100.0


def test_rouge_empty_inputs():
    """Empty prediction or reference should safely return 0.0 without errors."""
    evaluator = get_evaluator()
    scores = evaluator.compute_rouge("", "Some reference text.")
    assert scores["rouge1"] == 0.0
    assert scores["rougeL"] == 0.0

    scores = evaluator.compute_rouge("Some prediction text.", "")
    assert scores["rouge1"] == 0.0
    assert scores["rougeL"] == 0.0


def test_token_f1_calculation():
    """Verify word-level F1 token overlap calculation."""
    evaluator = get_evaluator()
    pred = "machine learning and deep learning"
    ref = "deep learning and neural networks"
    f1 = evaluator.compute_token_f1(pred, ref)
    assert 0.0 < f1 <= 100.0


def test_relative_delta_comparison():
    """Verify relative quality difference between baseline and optimized outputs."""
    evaluator = get_evaluator()
    base_out = "Artificial intelligence enables automated decision making."
    opt_out = "Artificial intelligence enables automated decision making in edge devices."
    ref = "Artificial intelligence enables automated decisions."

    delta_res = evaluator.compare_outputs(
        baseline_output=base_out,
        optimized_output=opt_out,
        reference_answer=ref
    )
    assert "delta_rouge1" in delta_res
    assert "delta_rougeL" in delta_res
    assert "output_agreement_f1" in delta_res


def test_compute_perplexity_from_nll():
    """Verify exponentiated perplexity mathematical formula."""
    evaluator = get_evaluator()
    import math

    losses = [1.0, 2.0, 3.0]
    expected = round(math.exp(2.0), 3)
    ppl = evaluator.compute_perplexity_from_nll(losses)
    assert ppl == expected


def test_compute_perplexity_with_mock():
    """Verify perplexity computation logic with a mock model and tokenizer."""
    import torch
    evaluator = get_evaluator()

    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            class Enc:
                input_ids = torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108]])
            return Enc()

    class MockOutputs:
        loss = torch.tensor(1.5)

    class MockModel:
        def __call__(self, input_ids, labels=None):
            return MockOutputs()

    res = evaluator.compute_perplexity(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        text="Sample evaluation text",
        device="cpu",
        max_length=4,
        stride=2,
    )

    assert "perplexity" in res
    assert "mean_loss" in res
    assert res["perplexity"] > 0
    assert res["num_chunks"] > 0

