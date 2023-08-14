from setuptools import setup, find_packages

setup(name='fbt',
      version='0.1',
      description='',
      url='',
      author='Sander_Broos',
      author_email='sander.broos.0@gmail.com',
      license='LICENSE.txt',
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'matplotlib',
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
