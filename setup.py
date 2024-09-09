from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name="voxws",
    version="1.0.1",
    description="Few Shot Language Agnostic Keyword Spotting (FSLAKWS) System",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Koushik S",
    author_email="koushik20040804@gmail.com",
    url="https://github.com/FewshotML/plix",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "wget",
        "librosa"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
    license="Apache-2.0",
    keywords=["Keyword Spotting", "Few-shot Learning", "Deep Neural Network", "Audio", "Speech"],
)