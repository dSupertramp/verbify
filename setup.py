from setuptools import setup, find_packages

setup(
    name="verbify",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-index"
        "llama-index-vector-stores-lancedb"
        "llama-index-llms-openai"
        "llama-index-llms-groq"
        "llama-index-llms-huggingface"
        "llama-index-embeddings-huggingface"
        "langchain"
        "python-dotenv"
        "requests"
        "nltk"
        "textblob"
    ],
    author="Danilo Palumbo",
    author_email="salvatoredanilopalumbo@gmail.com",
    description="Powered NLP with LLMs ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dsupertramp/verbify",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
