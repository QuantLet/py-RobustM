from setuptools import setup, find_packages

setup(
    name="py_robustm",
    version="0.0.0",
    # url="",
    author="Bruno Spilak",
    author_email="bruno.spilak@gmail.com",
    dependency_links=[],
    python_requires="~=3.8",
    install_requires=[
        "pandas==1.1.5",
        "rpy2==3.4.5",
        "cvxopt==1.2.7",
        "matplotlib==3.5.1",
        "scikit-learn==1.0.2",
        "scipy==1.7.3",
        "seaborn==0.11.2",
        "statsmodels==0.13.1",
        "joblib==1.1.0"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False,
    packages=find_packages()
)
