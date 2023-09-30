from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readline()
        requirements = [''.join(i.replace('\n', "") for i in requirements)]
        print(requirements)

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Shivam',
    author_email='rajputshivam.cse@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
