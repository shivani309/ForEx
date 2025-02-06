from setuptools import setup, find_packages

setup(
    name='swingtrading_indicators',
    version='0.1',
    description='A library for calculating technical indicators for swing trading',
    author=['Sejal Hanmante','Shivani Patil','Uzma patil'],
    author_email='sejal.hannmante@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
