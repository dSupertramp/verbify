from ..llm.base_llm import BaseLLM
from ..prompts.prompts import sentiment_analysis_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def sentiment_analysis(llm: BaseLLM):
    response_schemas = [
        ResponseSchema(
            name="text",
            description="Describes the text / sentence.",
        ),
        ResponseSchema(
            name="sentiment_label",
            description="Describes the sentiment label of the sentence.",
        ),
        ResponseSchema(
            name="sentiment_score",
            description="Describes the sentiment score of the sentence.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(query=sentiment_analysis_prompt, output_parser=output_parser)
    return response
