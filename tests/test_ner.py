import unittest
from unittest.mock import MagicMock
from verbify.ner import ner
from llm.base_llm import BaseLLM


class TestNER(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=BaseLLM)
        print("\nSetup: Created mock for BaseLLM")

    def test_ner(self):
        mock_response = [
            {
                "element": "Barack Obama",
                "type": "PERSON",
                "start_pos": 0,
                "end_pos": 12,
            },
            {"element": "Hawaii", "type": "LOCATION", "start_pos": 22, "end_pos": 28},
        ]
        self.mock_llm.query.return_value = mock_response
        print("\nTest NER: Set mock response for llm.query")
        print("Test NER: Calling ner function")
        result = ner(self.mock_llm)
        print("Test NER: Received result from ner function:", result)
        expected_result = mock_response
        print("Test NER: Expected result:", expected_result)
        self.assertEqual(result, expected_result)
        print("Test NER: Assertion passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
