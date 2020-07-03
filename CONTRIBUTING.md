# Contributing to xsec

## Repo structure

- The master branch is intepreted as the "latest, stable development version".

- All development is made on branches from master (or from other development branches). This is merged back to master only when stable and passing sanity checks.
 
- For every new feature release 1.x, we create a new branch "release_1.x" from master. This first commit on release_1.x gets the tag 1.x.0.
 
- Bugfixes for 1.x go directly on the release_1.x branch, with tags 1.x.1, 1.x.2, etc. (These tagged commits become the GitHub releases that we connect PyPI to.)

- When a new bugfix is implemented on release_1.x, we merge release_1.x --> master to propagate the fix to master, and from master onward to any development branch that may need it. (But we never merge master --> release_1.x.)
