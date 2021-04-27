# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	echo --- Install local package with tests in editable mode . . .
	cd .. && pip -q install -e .["tests"] # must install as  editable for autodocs to work
	echo --- Run package tests and generate coverage report . . .
	cd .. && pytest -n 4
	echo --- Run package style checks and generate style report . . .
	cd .. && pylint -n 4 -f text lind > ./docs/style/pylint_report.txt || echo "required to avoid makefile error"
	echo --- Run autodoc to update package documentation . . .
	sphinx-apidoc -f -o source/ ../lind
	echo --- Run documentation coverage rport . . .
	sphinx-build -b coverage . build/coverage
	echo --- logging current git tag . . .
	git describe --tags > ./latest_tag.txt
	echo --- Make documentation . . .
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
