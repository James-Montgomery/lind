# Project_Name

A short description.

### Authors

**James Montgomery** - *Initial Work* - [jamesmontgomery.us](http://jamesmontgomery.us)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

For a local installation, first git clone this repository. Then follow these instructions:

```
pip install .
```

To install from [pypi](#):

```
pip install -U project_name
```

To install the package with test dependencies add `["tests"]` to the install command:

```
pip install .["tests"]
# or
pip install -U project_name["tests"]
```

## Testing

Testing is an important part of creating maintainable, production grade code. Below are instructions for running unit and style tests as well as installing the necessary testing packages. Tests have intentionally been separated from the installable pypi package for a variety of reasons.

Make sure you have the required testing packages:

```
pip install -r requirements_test.txt
```

To install the project  with test dependencies see the install section.

### Running the unit tests

We use the pytest framework for unit testing.

```
pytest -vvl -s --cov project_name --disable-pytest-warnings
```

### Running the style tests

Having neat and legible code is important. Having documentation is also important. We use pylint as our style guide framework. Many of our naming conventions follow directly from the literary sources they come from. This makes it easier to read the mathematical equations and see how they translate into the code. This sometimes forces us to break pep8 conventions for naming.

```
pylint project_name --disable=invalid-name
```

## Contributor's Guide

Here are some basic guidelines for contributing.

### Branch Strategy

This repository doesn't use a complicated branching strategy. Simply create a feature branch off of master. When the feature is ready to be integrated with master, submit a pull request. A pull request will re quire at least one peer review and approval from the repository owner.

### Style Guide

Please stick to pep8 standards when for your code. Use numpy style docstrings.

### Test Requirements

Please use pytest as your testing suite. You code should have >= 80% coverage.

### Updating the Docs

<!--
sphinx-quickstart --ext-autodoc
# comment conf.py file
# add docs/.nojekyll file
# update build dir in docs/Makefile
# update static dir in docs/conf.py
# create dummy docs/index.html
-->

Updating the documentation is simple. First, let auto-docs check for updates to the package structure.

```
cd docs
sphinx-apidoc -o . ../project_name/
```

Finally, rebuild the html files.

```
make html
```

## Acknowledgments

People to acknowledge.

## TODO

Features to be added / change / fixed.

## Useful Resources

Useful resources for understanding the package.
