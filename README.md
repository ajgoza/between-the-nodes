# Between the Nodes

*A finite element story for AE 370 (Aerospace Numerical Methods, University of Illinois Urbana-Champaign).*

**Read the book:** <https://ajgoza.github.io/between-the-nodes/>

This repository holds the book's source: the chapters (`chapters/`), the Falkenrath-Voss thermal codes the characters use (`lib/`), and the styles. Every number and plot in the book is computed from `lib/` when the book is rendered.

## Running the characters' code

```
pip install https://github.com/ajgoza/between-the-nodes/archive/refs/heads/main.zip
```

then, in Python, `import falkenrath`.

## Building the book

Install [Quarto](https://quarto.org) and a Python with NumPy, SciPy, Matplotlib and Jupyter, then run `quarto render` in this folder. The site is written to `_book/`.

## License

Text and figures: CC BY-NC 4.0. Code: MIT. See `LICENSE.md`.
