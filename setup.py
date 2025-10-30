from setuptools import setup, find_packages

setup(
    name="clipsex",
    version="0.1.0",
    description="PhotoMind - Local semantic photo search with CLIP",
    author="Kairav Deepeshwar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "open-clip-torch>=2.20.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "faiss-cpu>=1.7.4",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
    },
)
