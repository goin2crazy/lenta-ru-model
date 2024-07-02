from setuptools import setup, find_packages

setup(
    name="awesome-inference",
    version="0.1.0",
    author="Double Cringe",
    author_email="yanyap666@gmail.com",
    description="A package for awesome inference with transformers, torch, and numpy",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "datasets"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'awesome-inference=main:Inference',
        ],
    },
)
