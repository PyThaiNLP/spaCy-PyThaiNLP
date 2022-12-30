from setuptools import find_packages, setup

requirements = [
    "pythainlp>=3.1.0",
    "spacy>=3.0"
]

with open("README.md", "r") as f:
    readme = f.read()


setup(
    name="spacy-pythainlp",
    version="0.1dev2",
    description="PyThaiNLP For spaCy",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong Phatthiyaphaibun",
    author_email="wannaphong@yahoo.com",
    url="https://github.com/PyThaiNLP/spaCy-PyThaiNLP",
    packages=["spacy_pythainlp"],
    # test_suite="tests",
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
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
