from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="voxws",
    version="1.0.1",
    description="Few Shot Language Agnostic Keyword Spotting (FSLAKWS) System",
    long_description_content_type="text/markdown",
    long_description=README,
    license="Apache-2.0",
    packages=['voxws'],
    author="Koushik S",
    author_email="koushik20040804@gmail.com",
    keywords=["Keyword Spotting", "Few-shot Learning", "Deep Neural Network", "Audio", "Speech"],
    url="https://github.com/FewshotML/plix",
    download_url="https://pypi.org/project/voxws/",
)

install_requires = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "wget",
    "librosa"
]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)