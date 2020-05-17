from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vol4life',
    version='0.0.1',
    description='Finance tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Felipe Salvatore',
    author_email='felipessalvador@gmail.com',
    packages=find_packages(),
    license="Apache License, Version 2.0",
    test_suite="tests",
    url="https://github.com/felipessalvatore/vol4life",
    keywords=["finance", "quant"],
        classifiers=[
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
)