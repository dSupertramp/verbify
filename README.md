# Verbify

Verbify is an open-source library that integrates Large Language Models (LLMs) with traditional NLP libraries to provide advanced text analysis and generation capabilities.

## Features

- **Advanced Sentiment Analysis**
- **Contextual Text Generation**
- **Enhanced Named Entity Recognition (NER)**
- **Text Summarization**
- **Text Classification**
- **Question Answering**
- **Grammar and Style Correction**
- **Language Translation**

And so on

## 📦 Install

First of all, create a virtual environment with the command `python3 -m venv venv_name` and activate it with `source venv_name\bin\activate`.

After that, install the required libraries with `pip install -r requirements.in`

## 🔑 Set API Keys

In order to set API keys, insert your keys into the `env.example` file and rename it to `.env`.

## 🔍 Usage

Instantiate an LLM with your text:

```python
text = "The latest iPhone model has many new features."
llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"), text=text)
```

And extract informations:

```python
print(sentiment_analysis(llm=llm))
```

```python
{
  'text': 'The latest iPhone model has many new features.',
  'sentiment_label': 'Positive',
  'sentiment_score': '0.8'
  }
```

```python
print(translate(llm=llm, language="it"))
```

```python
{
  'text': 'The latest iPhone model has many new features.',
  'translated_text': "L'ultimo modello di iPhone ha molte nuove funzionalità."
}
```

```python
print(classify(llm=llm, language="it"))
```

```python
{
  'text': 'The latest iPhone model has many new features.',
  'labels': ['Technology', 'Gadgets', 'Innovation', 'Electronics', 'Smartphones']
}
```
