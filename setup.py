import os

from setuptools import find_packages, setup


def read_requirements():
    """Read install requirements from requirements.txt (single source of truth)."""
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, "requirements.txt")
    requirements = []
    if not os.path.exists(req_path):
        return requirements
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            requirements.append(line)
    return requirements


setup(
    name='audiox-turbo',
    version='0.1.0',
    url='https://github.com/NoizAI/AudioX-Turbo',
    author='AudioX-Turbo contributors',
    description='AudioX-Turbo four-step DMD and GAN distillation training code',
    packages=find_packages(),
    install_requires=read_requirements(),
)
