from setuptools import setup, find_packages

setup(
    name="atometric",
    description="A metric for evaluating the quality of atomic statements in a text given reference information",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "evaluate",
        "langchain",
        "pydantic",
    ],
    include_package_data=True,
    package_data={
        "atometric": ["assets/**/*"],
    },
)
