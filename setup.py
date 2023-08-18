from setuptools import setup, find_packages
import sys

from Cython.Build import cythonize
USE_CYTHON = True

try:
    import numpy as np
except ImportError:
    sys.exit("Could not import numpy, which is required to build the extension modules.")

extensions = cythonize("grad_tts/model/monotonic_align/*.pyx")
for ext_module in extensions:
    ext_module.include_dirs.append(np.get_include())

setup(
    name='grad_tts',
    version='0.0.1',
    packages=find_packages(),
    package_data = {
        '': ['cmu_dictionary', 'hifigan-config.json']
    },
    ext_modules = cythonize(extensions),
    author='Li Wang',
    author_email='li@liwang.info',
    description='A fork of the official implementation of the Grad-TTS model.',
    url='https://github.com/liwangd/Grad-TTS',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    setup_requires=["Cython", "numpy"],
    install_requires=["Cython", "numpy"],
)
