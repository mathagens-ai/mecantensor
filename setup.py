from setuptools import setup, find_packages

setup(
    name="mecantensor",
    version="2.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mecantensor": ["*.dll", "*.so", "*.pyd", "../src/hal/*.py", "../src/midbits/*.py"],
    },
)
