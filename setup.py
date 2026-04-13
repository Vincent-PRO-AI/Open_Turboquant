from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path) as f:
        long_description = f.read()

setup(
    name="turboquant",
    version="1.0.0",
    description="TurboQuant: KV Cache Compression for LLMs (ICLR 2026) + PolarQuant (AISTATS 2026)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vincent Soule",
    author_email="vincent.soule@arkanecloud.com",
    url="https://github.com/vincentsoule/turboquant",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "accelerate>=0.26.0",
    ],
    extras_require={
        "triton": ["triton>=2.2.0"],
        "dev": ["pytest>=7.0", "triton>=2.2.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache 2.0",
    keywords="llm quantization kv-cache compression inference triton",
)
