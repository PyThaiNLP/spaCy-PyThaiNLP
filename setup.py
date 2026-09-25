from setuptools import find_packages, setup

requirements = [
    "pythainlp>=3.1.0",
    "spacy>=3.0",
    "gensim>=4.0",
    "python-crfsuite"
]

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()


setup(
    name="spacy-pythainlp",
    version="1.1.0",
    description="PyThaiNLP For spaCy",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    url="https://github.com/PyThaiNLP/spaCy-PyThaiNLP",
    packages=find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    entry_points={
        "spacy_factories": [
            "pythainlp = spacy_pythainlp.core:PyThaiNLP",
            "pythainlp_sentencizer = spacy_pythainlp.components:PyThaiNLPSentencizer",
            "pythainlp_tagger = spacy_pythainlp.components:PyThaiNLPTagger",
            "pythainlp_ner = spacy_pythainlp.components:PyThaiNLPNER",
            "pythainlp_parser = spacy_pythainlp.components:PyThaiNLPParser",
            "pythainlp_vectors = spacy_pythainlp.components:PyThaiNLPVectors",
            "pythainlp_lemmatizer = spacy_pythainlp.components:PyThaiNLPLemmatizer",
        ],
        "spacy_tokenizers": [
            "pythainlp_tokenizer = spacy_pythainlp.tokenizer:create_pythainlp_tokenizer",
        ],
    },
    keywords=[
        "pythainlp",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "ThaiNLP",
        "Thai NLP",
        "Thai language",
        "spacy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: Thai",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Source Code": "https://github.com/PyThaiNLP/spaCy-PyThaiNLP",
        "Bug Tracker": "https://github.com/PyThaiNLP/spaCy-PyThaiNLP/issues",
    },
)
