To generate the docs: 

## Install packages

```
pip install quarto
pip install quartodoc
quarto add machow/quartodoc
```

## Build

Pages and figures are handled separately. Figures do not get regenerated on build, nor do notebooks get rerun. 

To rebuild the pages:
```
cd docs
python scripts/sync_notebooks.py
python scripts/generate_tutorials_index.py
quartodoc build
quarto render
```

This will produce a _site folder, and you can inspect the built documentation by opening `index.html` in a browser.


To regenerate the figures:
```
python scripts/make_doc_figures.py
```

Notebooks will be added as-is. 

## Deploy
Site will be built and deployed automatically via github actions when changes are pushed to `main` or a release is made. Additionally, a build can be triggered manually via github actions. 