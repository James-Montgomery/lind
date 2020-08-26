# Lind

This package is meant to house common tools used in experiments design and analysis. There is little "novel" functionality here, but it is hopefully packaged in a way that is convenient and useful for users.

I began this package for two reasons. First, I was  dissatisfied with the existing packages available for experimentation. They seemed like collections of random tools rather than a cohesive set of utilities that work together in harmony to a united purpose. Second, I used this as an opportunity to refresh my understanding of various statistical tools and methods.

### Authors

**James Montgomery** - *Initial Work* - [jamesmontgomery.us](http://jamesmontgomery.us)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

### About the Name

Looking for a name for this package I tried looking back into the history of experimentation. I was tempted to name the package after King Nebuchadnezzar in reference to the "legumes and water" anecdote from the book of Daniel. This is often considered one of the earliest controlled "trials".

However, some of the first modern controlled trials were conducted by Dr. James Lind. There are many scatter references to trials throughout history, but Lind represented the start of the modern era of controlled trials and their integration into the scientific method. Hence I named the package after Lind. If you  have a chance, I recommend taking an afternoon and reading about the work Lind did to fight the disease Scurvy.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

For a local installation, first git clone this repository. Then follow these instructions:

```
pip install .
```

To install from [pypi](https://pypi.org/project/lind/):

```
pip install -U lind
```

To install the package with test dependencies add `["tests"]` to the install command:

```
pip install .["tests"]
# or
pip install -U lind["tests"]
```

## Testing

Testing is an important part of creating maintainable, production grade code. Below are instructions for running unit and style tests as well as installing the necessary testing packages. Tests have intentionally been separated from the installable pypi package for a variety of reasons.

Make sure you have the required testing packages:

```
pip install -r requirements_test.txt
```

To install the project  with test dependencies see the install section.

### Running the unit tests

We use the pytest framework for unit testing. Test preset args are defined in `pytest.ini`.

```
pytest
```

### Running the style tests

Having neat and legible code is important. Having documentation is also important. We use pylint as our style guide framework. Many of our naming conventions follow directly from the literary sources they come from. This makes it easier to read the mathematical equations and see how they translate into the code. This sometimes forces us to break pep8 conventions for naming. Linting presets are defined in pylintrc.

```
pylint lind
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

Updating the documentation is simple. First, let auto-docs check for updates to the package structure.

```
cd docs
make html
```

## Acknowledgments

People to acknowledge.

## TODO

Features to be added / change / fixed.

## Useful Resources

Useful resources for understanding the package.
