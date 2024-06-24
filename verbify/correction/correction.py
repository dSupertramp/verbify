from ..llm.base_llm import BaseLLM
from ..prompts.prompts import correction_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def correct(llm: BaseLLM):
    response_schemas = [
        ResponseSchema(
            name="text",
            description="Describes the text / sentence.",
        ),
        ResponseSchema(
            name="corrected_text",
            description="Describes the corrected text.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(query=correction_prompt, output_parser=output_parser)
    return response
