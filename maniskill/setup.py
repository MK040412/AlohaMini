from setuptools import setup, find_packages

setup(
    name="aloha_mini_maniskill",
    version="0.1.0",
    description="AlohaMini dual-arm mobile robot for ManiSkill3",
    author="AlohaMini Team",
    packages=find_packages(),
    package_data={
        "aloha_mini": [
            "*.urdf",
            "meshes/*.STL",
        ],
    },
    include_package_data=True,
    install_requires=[
        "mani-skill>=3.0.0b0",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pygame",  # For keyboard control
        ],
    },
    python_requires=">=3.8",
)
