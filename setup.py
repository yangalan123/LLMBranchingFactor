from setuptools import setup, find_packages

setup(
    name='branching_factor',
    version='0.1',
    package_dir={"uncertainty_quantification": "."},
    packages=find_packages(where="."),
    url='',
    license='',
    author='Chenghao Yang',
    author_email='chenghao@uchicago.edu',
    description='Branching Factor Evaluation for LLMs',
)