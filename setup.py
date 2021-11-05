from setuptools import setup, find_packages

setup(
    name="py_RobustM",
    version="0.0.0",
    # url="",
    author="Bruno Spilak",
    author_email="bruno.spilak@gmail.com",
    dependency_links=[],
    python_requires="~=3.7",
    install_requires=[
        "rpy2==3.4.5",
        "cvxopt==1.2.7",
        "matplotlib==3.4.3",
        "pandas==1.3.4",
        "scikit-learn==1.0.1",
        "scipy==1.7.1",
        "seaborn==0.11.2",
        "statsmodels==0.13.0",
        "joblib==1.1.0"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False,
    packages=find_packages()
)
