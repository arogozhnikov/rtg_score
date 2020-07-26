from setuptools import setup

setup(
    name="rtg_score",
    version='0.1.0',
    description="Analysis of confounders by Rank-to-Group scores",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Alex Rogozhnikov, System1 Biosciences',
    packages=['rtg_score'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],
    keywords='variability analysis, variability decomposition, contributing factors',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
    ],
)