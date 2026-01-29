import pytest
import mock
import numpy as np

from coral_patterns.dla_parameterized import normalize_probabilities

# Sample test file because git doesn't allow empty directories

def test_normalize_probabilities():
    input_data = np.array([1, 2, 3, 4])
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    result = normalize_probabilities(input_data)
    assert np.allclose(result, expected)

