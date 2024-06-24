from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseLLM(ABC):
    """
    Base LLM class.
    """

    @abstractmethod
    def query(self, query: str, output_parser: Optional[Any]) -> str:
        """
        Query method for LLM.

        Args:
            query (str): Query
            output_parser (Optional[Any]): Output parser to parse the response

        Returns:
            str: LLM response
        """
        pass
