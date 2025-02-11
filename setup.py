from setuptools import setup,find_packages
setup(
    name="rxngraphormer",
    version="0.6.0",
    description="Package for a novel graph-based transformer model for reaction prediction",
    classifiers=["Development Status :: 1 - Pre-Alpha", "Topic :: Scientific/Engineering :: Chemistry"],
    keywords=[],
    url="https://github.com/licheng-xu-echo/RXNGraphormer",
    author="Li-Cheng Xu",
    author_email="xulicheng@sais.com.cn",
    license="MIT License",
    packages=find_packages(),
    install_package_data=True,
    zip_safe=False,
    install_requires=[],
    package_data={"":["*.csv"]},
)
