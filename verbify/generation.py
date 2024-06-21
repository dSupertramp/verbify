from llm.base_llm import BaseLLM
from prompts.prompts import contextual_text_generation_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def contextual_text_generation(llm: BaseLLM, max_length: int):
    response_schemas = [
        ResponseSchema(
            name="text",
            description="Describes the text / sentence.",
        ),
        ResponseSchema(
            name="complete_text",
            description="Describes the text with the new generation.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(
        query=contextual_text_generation_prompt.format(max_length=max_length),
        output_parser=output_parser,
    )
    return response
