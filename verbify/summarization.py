from llm.base_llm import BaseLLM
from prompts.prompts import summarize_prompt
from llama_index.core.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def summarize(llm: BaseLLM):
    response_schemas = [
        ResponseSchema(
            name="summary",
            description="Describes the summary of the text / sentence.",
        ),
    ]
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    response = llm.query(query=summarize_prompt, output_parser=output_parser)
    return response
