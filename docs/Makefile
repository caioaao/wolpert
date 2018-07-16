# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = wolpert
SOURCEDIR     = source
BUILDDIR      = build
MODULEDIR     = ../wolpert

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	-rm -rf "${SOURCEDIR}/generated"
	-rm -rf "${BUILDDIR}/html"

apidoc:
	SPHINX_APIDOC_OPTIONS=inherited-members sphinx-apidoc -o "${SOURCEDIR}/generated/" "${MODULEDIR}" '../wolpert/sklearn_pipeline.py' '../**/tests/*'

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: apidoc Makefile 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
