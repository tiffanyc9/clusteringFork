import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustering-nassir",
    version="0.1.1",
    author="Nassir Mohammad",
    author_email="drnassirmohammad@gmail.com",
    description="A semi-supervised clustering method using anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m-nassir/clustering",
    packages=setuptools.find_packages(include=["clustering_nassir", "clustering_nassir.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    "numpy",
    "perception_nassir",
    ],
    python_requires=">=3.8",
    license="MIT",
)
