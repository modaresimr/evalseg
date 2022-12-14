[![pypi](https://img.shields.io/pypi/v/evalseg.svg)](https://pypi.org/project/evalseg)
[![codecov](https://codecov.io/gh/modaresimr/evalseg/branch/main/graph/badge.svg?token=evalseg_token_here)](https://codecov.io/gh/modaresimr/evalseg)
[![CI](https://github.com/modaresimr/evalseg/actions/workflows/main.yml/badge.svg)](https://github.com/modaresimr/evalseg/actions/workflows/main.yml)

# Medical Image Segmentation Evaluation

This project is intended to evaluate Medical Segmentation approaches from multiple prespective.

Please cite us if you use our codes or ideas ;)

```
@misc{modaresi2022mis,
  doi = {10.34740/KAGGLE/DSV/4728423},
  url = {https://www.kaggle.com/dsv/4728423},
  author = {{Seyed Modaresi} and {Aomar Osmani} and {Mohammadreza Razzazi} and {Abdelghani Chibani}},
  title = {Medical Image Segmentation: Evaluation},
  publisher = {Kaggle},
  year = {2022}
}
```
![graphic_abstract](https://user-images.githubusercontent.com/9498182/200766838-d133d84f-2805-4818-b98c-a470debdb6ee.png)

![graphic-abstract](https://user-images.githubusercontent.com/9498182/197723220-760ad148-e7a7-4bd6-bacb-805111141dcc.png)

![3d](https://user-images.githubusercontent.com/9498182/200767060-946e7184-7d3c-447f-8df1-5edc195e4c0f.png)

### What is included on this repository?

- ๐ Documentation
- ๐ A simple [Containerfile](Containerfile) to build a container image for your project.  
  `Containerfile` is a more open standard for building container images than Dockerfile, you can use buildah or docker with this file.
- ๐งช Testing structure using [pytest](https://docs.pytest.org/en/latest/)
- โ Code linting using [flake8](https://flake8.pycqa.org/en/latest/)
- ๐ Code coverage reports using [codecov](https://about.codecov.io/sign-up/)
- ๐ณ๏ธ Automatic release to [PyPI](https://pypi.org) using [twine](https://twine.readthedocs.io/en/latest/) and github actions.
- ๐ฏ Entry points to execute your program using `python -m <evalseg>` or `$ evalseg` with basic CLI argument parsing.
- ๐ Continuous integration using [Github Actions](.github/workflows/) with jobs to lint, test and release your project on Linux, Mac and Windows environments.

---

# evalseg

## Install it from PyPI

```bash
pip install evalseg
```

```bash
pip install git+https://github.com/modaresimr/evalseg
```

## Usage

```py
comming soon
```

```bash
$ python -m evalseg
#or
$ evalseg
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
