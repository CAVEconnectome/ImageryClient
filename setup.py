from setuptools import setup
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('short_readme.md', 'r') as f:
    long_description = f.read()

setup(
    version=find_version("imageryclient", "__init__.py"),
    name='imageryclient',
    description='Front end tools for composite images for EM connectomics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Casey Schneider-Mizell',
    author_email='caseysm@gmail.com',
    url='https://github.com/ceesem/ImageryClient',
    packages=['imageryclient'],
    install_requires=required,
    setup_requires=['pytest-runner'],
)
