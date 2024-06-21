from setuptools import setup, find_packages

setup(
    name="verbify",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "textblob",
        "transformers",
    ],
    author="Danilo Palumbo",
    author_email="salvatoredanilopalumbo@gmail.com",
    description="An NLP library integrating LLMs for advanced text processing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dsupertramp/verbify",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
