# Template Instructions

Instructions for how to use this repository template. I assume users of this
template are already familiar with python and Github. The intent of this
template is to save time, not teach.

## Instructions

#### Step One - Create New  Repository

Create a new repository using these template as the base. Follow instructions
[HERE](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template).
You should name your repository after the name of your project. For example, if
I was building a python tool called `Steve`, I would name the repository `Steve`.

#### Step Two - Renaming

The repository is currently set up for a project called `project_name`. You will
need to update this to your actual project name in various places. This should
be easy using a text editor like Atom or an IDE like Pycharm.

This template was also created primarily for personal use. This means that
`./LICENSE.md` and `./docs/conf.py` reference my name for license and copywrite.
You will need to replace my name with yours.

You will end up rewriting the contents of the `./tests` and `project_name`
directories, so don't worry about editing the files in these directories.

Places to update name:
1. Update name of directory `./project_name`
2. Update names in `setup.py`
3. Update names in `setup.cfg`
4. Update names in `./docs/index.rst`
4. Update names in `./docs/conf.py`
5. Update names in `./README.md`
6. Update names in `./LICENSE.md`

#### Step Three - Dependencies

You should have all of the dependencies you need for testing and documentation
defined in `requirements_test.txt` and `requirements_docs.txt`. For more advanced
testing use cases you may need to add dependencies to these files. We assume
that you are using pytests.

You will need to replace the package dependencies in `requirements.txt` with the
dependencies for your own project.

#### Step Four - Establish Well  Managed Practices

For a well managed project, it is important to add protections to your branches
to ensure proper workflows from contributors. First, navigate to `settings`
for your repository in the Github web GUI. Go to `Branches` and add branch
specific protections in line with your choice of branching strategy
(i.e. Gitflow).

I recommend looking at the protections / rules:
* Require pull request reviews before merging
* Require status checks to pass before merging (if using a CICD pipeline like Jenkins)
* Include administrators

If you are using a CICD tool, remember to set up the appropriate webhooks to
your pipeline.

Finally, go back to `settings` for your repository in the Github web GUI.
Scroll down to `Merge button`. Uncheck all options except
`Allow squash merging`. Then,  below, check
`Automatically delete head branches`. These changes will help keep your merge
history and PR requests clean and clear.

#### Step Five - Create Your Package

Replace the code in `./project_name` (which you should have renamed to your
project name) and `./tests` with your project code and tests.

This is the fun part, the actual coding! Remember to add proper docstrings as
they will be used for automated  document generation in the next step. Also,
we recommend to add thorough unit, integration, and smoke  testing for your
package.

Depending on your package set up, you may need  to update the `./setup.py` file
to ensure reliable installation of your package. For example, if you use fortran
or C++ extensions in your project / package / library.

#### Step Six - Write Documentation

Assuming you have followed the steps above properly,jsut run `make html` inside
of `./docs` to create your documents. You will need to do this every time you
update your code.

To display your documentation in a gitpages website, navigate to `settings`
for your repository in the Github web GUI. Scroll down to `Github Pages`.
Under `Source` select `master branch /docs folder`.

#### Step Seven - README

Open the `./README.md` file. Filling in the required details as required. It
should be self explanatory.

#### Step Eight - Finish Up

If you haven't already been, go ahead and push all of this to github. Your
package should be ready to use!

## Notes

* We highly recommend using controlled working environments for develipment such
as `conda virtual environments` or `docker containers`.
* We recommend using proper text editors and IDEs for development such as Atom
and Pycharm
* We highly recommend `numpy style docstrings`
* We highly recommend assing a description, website url, and topics to your project
Github repository to make it easy to find for users and contributors
* We highly recommend making your package available for use through pypi  
