"""Tests for concept_extraction_v2.py — semantic concept extraction pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from concept_extraction_v2 import (
    ExtractionState,
    EpisodeConcepts,
    format_conversation_for_extraction,
    build_extraction_prompt,
    parse_concept_list,
    extract_concepts_from_episode,
    get_remaining_episodes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_conversation():
    return {
        'title': 'MCMC in Python',
        'conversation_id': 'abc123',
        'model': 'gpt-4',
        'messages': [
            {'role': 'user', 'content': 'How do I implement Metropolis-Hastings in Python?'},
            {'role': 'assistant', 'content': 'The Metropolis-Hastings algorithm is a Markov chain Monte Carlo method...'},
            {'role': 'user', 'content': 'Can you show me the acceptance ratio calculation?'},
            {'role': 'assistant', 'content': 'The acceptance ratio alpha = min(1, p(x_new)/p(x_old) * q(x_old|x_new)/q(x_new|x_old))...'},
        ],
    }


@pytest.fixture
def empty_state():
    return ExtractionState()


@pytest.fixture
def state_with_vocab():
    state = ExtractionState()
    state.vocabulary = ['Bayesian inference', 'Python programming', 'MCMC sampling']
    return state


# ---------------------------------------------------------------------------
# Tests: conversation formatting
# ---------------------------------------------------------------------------

def test_format_conversation_basic(sample_conversation):
    text = format_conversation_for_extraction(sample_conversation)
    assert 'MCMC in Python' in text
    assert '[user]' in text
    assert 'Metropolis-Hastings' in text


def test_format_conversation_truncation():
    doc = {
        'title': 'Long Conv',
        'messages': [
            {'role': 'user', 'content': 'x' * 1000}
            for _ in range(50)
        ],
    }
    text = format_conversation_for_extraction(doc, max_chars=500)
    assert len(text) <= 600  # some slack for title/formatting


def test_format_conversation_list_content():
    """Handle messages with list-format content."""
    doc = {
        'title': 'Test',
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Hello world'},
                {'type': 'text', 'text': 'More text'},
            ]},
        ],
    }
    text = format_conversation_for_extraction(doc)
    assert 'Hello world' in text


def test_format_conversation_empty_messages():
    doc = {'title': 'Empty', 'messages': []}
    text = format_conversation_for_extraction(doc)
    assert 'Empty' in text


# ---------------------------------------------------------------------------
# Tests: prompt building
# ---------------------------------------------------------------------------

def test_prompt_without_vocabulary(sample_conversation):
    text = format_conversation_for_extraction(sample_conversation)
    prompt = build_extraction_prompt(text, [])
    assert 'EXISTING CONCEPTS' not in prompt
    assert '3-10 concepts' in prompt
    assert 'noun phrase' in prompt


def test_prompt_with_vocabulary(sample_conversation):
    text = format_conversation_for_extraction(sample_conversation)
    vocab = ['Bayesian inference', 'MCMC sampling']
    prompt = build_extraction_prompt(text, vocab)
    assert 'EXISTING CONCEPTS' in prompt
    assert 'Bayesian inference' in prompt
    assert 'MCMC sampling' in prompt


def test_prompt_large_vocabulary_truncated(sample_conversation):
    text = format_conversation_for_extraction(sample_conversation)
    vocab = [f'concept_{i}' for i in range(500)]
    prompt = build_extraction_prompt(text, vocab, max_vocab_shown=100)
    assert 'concept_0' in prompt
    assert 'concept_99' in prompt
    assert '400 more' in prompt


# ---------------------------------------------------------------------------
# Tests: response parsing
# ---------------------------------------------------------------------------

def test_parse_json_array():
    response = '["Bayesian inference", "MCMC sampling", "Python numerical computing"]'
    concepts = parse_concept_list(response)
    assert len(concepts) == 3
    assert 'Bayesian inference' in concepts


def test_parse_json_in_text():
    response = 'Here are the concepts:\n["concept one", "concept two"]\nDone.'
    concepts = parse_concept_list(response)
    assert len(concepts) == 2


def test_parse_empty_array():
    response = '[]'
    concepts = parse_concept_list(response)
    assert concepts == []


def test_parse_quoted_fallback():
    response = 'The concepts are "machine learning" and "neural networks".'
    concepts = parse_concept_list(response)
    assert len(concepts) >= 2
    assert 'machine learning' in concepts


def test_parse_filters_empty_strings():
    response = '["good concept", "", "  ", "another good one"]'
    concepts = parse_concept_list(response)
    assert len(concepts) == 2


# ---------------------------------------------------------------------------
# Tests: extraction state
# ---------------------------------------------------------------------------

def test_state_add_updates_vocabulary(empty_state):
    result = EpisodeConcepts(
        episode_id='test-1',
        title='Test',
        concepts=['Bayesian inference', 'MCMC sampling'],
        message_count=4,
        extraction_time=1.0,
    )
    empty_state.add(result)

    assert len(empty_state.vocabulary) == 2
    assert 'Bayesian inference' in empty_state.vocabulary
    assert empty_state.total_extractions == 1
    assert empty_state.total_concepts_raw == 2


def test_state_deduplicates_vocabulary(empty_state):
    r1 = EpisodeConcepts('ep1', 'T1', ['Bayesian inference', 'MCMC'], 4, 1.0)
    r2 = EpisodeConcepts('ep2', 'T2', ['bayesian inference', 'Neural networks'], 4, 1.0)

    empty_state.add(r1)
    empty_state.add(r2)

    # "bayesian inference" should not be duplicated (case-insensitive)
    assert len(empty_state.vocabulary) == 3
    assert empty_state.total_concepts_raw == 4


def test_state_save_load(empty_state, tmp_path):
    r1 = EpisodeConcepts('ep1', 'Test', ['concept A', 'concept B'], 4, 1.0)
    empty_state.add(r1)

    state_file = tmp_path / "state.json"
    empty_state.save(state_file)

    loaded = ExtractionState.load(state_file)
    assert loaded.total_extractions == 1
    assert len(loaded.vocabulary) == 2
    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].episode_id == 'ep1'


# ---------------------------------------------------------------------------
# Tests: end-to-end with mock LLM
# ---------------------------------------------------------------------------

def test_extract_concepts_mock_llm(tmp_path):
    """Test full extraction with a mock LLM."""
    # Create a fake conversation
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    doc = {
        'title': 'Test Conversation',
        'messages': [
            {'role': 'user', 'content': 'How does gradient descent work?'},
            {'role': 'assistant', 'content': 'Gradient descent is an optimization algorithm...'},
        ],
    }
    with open(conv_dir / "test-conv.json", 'w') as f:
        json.dump(doc, f)

    state = ExtractionState()

    def mock_llm(prompt):
        return '["Gradient descent", "Optimization algorithms", "Machine learning"]'

    result = extract_concepts_from_episode(
        'test-conv', state, mock_llm, conv_dir
    )

    assert result.episode_id == 'test-conv'
    assert result.title == 'Test Conversation'
    assert len(result.concepts) == 3
    assert 'Gradient descent' in result.concepts
    assert len(state.vocabulary) == 3


def test_remaining_episodes(tmp_path):
    """Test that remaining episodes excludes already-extracted ones."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    for name in ['ep-a', 'ep-b', 'ep-c']:
        with open(conv_dir / f"{name}.json", 'w') as f:
            json.dump({'title': name, 'messages': []}, f)

    state = ExtractionState()
    state.episodes.append(EpisodeConcepts('ep-a', 'A', ['x'], 1, 0.5))

    remaining = get_remaining_episodes(state, conv_dir)
    assert 'ep-a' not in remaining
    assert 'ep-b' in remaining
    assert 'ep-c' in remaining
