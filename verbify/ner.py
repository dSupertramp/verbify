from llm.base_llm import BaseLLM
from prompts.prompts import ner_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def ner(llm: BaseLLM):
    response_schemas = [
        ResponseSchema(
            name="element",
            description="Describes the element",
        ),
        ResponseSchema(
            name="type",
            description="Describes the type of the element.",
        ),
        ResponseSchema(
            name="start_pos",
            description="Describes the start position (characters) of the element.",
        ),
        ResponseSchema(
            name="end_pos",
            description="Describes the end position (characters) of the element.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(query=ner_prompt, output_parser=output_parser)
    return response
