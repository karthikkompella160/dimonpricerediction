from setuptools import setup,find_packages
from typing import List


HYPEN_E_DOT="-e ."

def get_requirements(filename:str)->List[str]:
    requirements=[]
    with open(filename) as file_obj:
       requirements= file_obj.readlines()
       requirements=[req.replace("\n","") for req in requirements]
       if(HYPEN_E_DOT in requirements):
           requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="Dimond price prediction",
    version="0.0.1",
    author="Karthik kompella",
    author_email="karthikkompella160@gmail.com",

    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)