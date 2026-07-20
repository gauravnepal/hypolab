"""Setup script for HypoLab."""

from setuptools import setup, find_packages

setup(
    name="hypolab",
    version="0.1.0",
    author="Gaurav Nepal",
    description="Agentic Data Pipeline for Automated Hypothesis Generation & Testing",
    long_description="HypoLab: Automated hypothesis generation, statistical validation, and literature search powered by LLMs.",
    long_description_content_type="text/plain",
    url="https://github.com/gauravnepal/hypolab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff>=0.1.0"],
        "app": ["streamlit>=1.28.0"],
        "groq": ["groq>=0.4.0"],
        "local": ["transformers>=4.35.0", "torch>=2.0.0", "accelerate>=0.24.0"],
    },
)