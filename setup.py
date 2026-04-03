from setuptools import setup, find_packages

setup(
    name="tq-impl",
    version="2.0.0",
    description="TurboQuant: Near-Optimal KV Cache Compression for LLMs (ICLR 2026)",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Vincent Soule",
    url="https://github.com/vincentsoule/tq-impl",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "scipy",
    ],
    extras_require={
        "triton": ["triton>=2.1"],
        "hf": ["transformers>=4.40", "accelerate"],
        "dev": ["pytest", "triton>=2.1", "transformers>=4.40", "accelerate"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache-2.0",
)
