from setuptools import setup, find_packages
from typing import List

EPHINE_DOT_E = '-e .'


def get_requirements(file_name: str) -> List[str]:
    try:
        requirements = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if line != EPHINE_DOT_E:
                    requirements.append(line)
        return requirements
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found.")



setup(
    name='Predict_a_user_future_coordinates',
    version='0.0.1',
    author='Vikrant',
    author_email='patilvikrant275@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)
