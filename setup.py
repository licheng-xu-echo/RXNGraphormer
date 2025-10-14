import os
from setuptools import setup,find_packages
def read_requirements():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(base_dir, 'requirements.txt')
    with open(req_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]
setup(
    name="rxngraphormer",
    version="1.0.1",
    description="Package for a novel graph-based transformer model for reaction prediction",
    keywords=[],
    url="https://github.com/licheng-xu-echo/RXNGraphormer",
    author="Li-Cheng Xu",
    author_email="xulicheng@sais.com.cn",
    license="MIT License",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=read_requirements(),
    package_data={"":["*.csv"]},
)
