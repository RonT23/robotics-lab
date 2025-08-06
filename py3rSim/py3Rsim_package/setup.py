from setuptools import setup, find_packages

setup(
    name='py3Rsim_package',
    version='0.1',
    author='Ronaldo Tsela',
    author_email='rontsela@mail.ntua.gr',
    description='A simple 3R robot manipulator simulator for trajectory design and tracking with animation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/RonT23/py3Rsim/py3Rsim_package',
    
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.5',
    
    install_requires=[  
        'matplotlib'
    ],
)