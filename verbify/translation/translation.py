from ..llm.base_llm import BaseLLM
from ..prompts.prompts import translate_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def translate(llm: BaseLLM, language: str):
    response_schemas = [
        ResponseSchema(
            name="text",
            description="Describes the text / sentence.",
        ),
        ResponseSchema(
            name="translated_text",
            description="Describes the translated text.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(
        query=translate_prompt.format(language=language), output_parser=output_parser
    )
    return response
