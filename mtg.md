## Meeting notes

### Next meeting: TBD (end of May/beginning of June?)

#### Topics for next meeting



#### To do

- [ ] Getting Jeriek up to speed with (D)GPs and using Prospino (Jeriek, Ingrid and Jon Vegard)
- [ ] Check what is the minimal amount of storage required to recreate the GPs (Jeriek and Ingrid)

* [ ] Look into GAMBIT interfacing (Anders)
* [ ] Look into including PDF uncertainties (Are)



---

### 30.04.2018 (first meeting!) 

##### 1. Project name

Currently XSEC (or TUXEDO, The Universal X-section EstimaDOr, code name SMOKING), no final decision was made about the project name. A mythological reference would be nice (but no, Tanngnjóstr :goat: is too hard). Preferably a name that tab-completes easily in a terminal.

Some other ideas:

* FACET(S): Fast and Accurate Cross-section Estimation Tool (for SUSY) 
* ACES: Accelerated Cross-section Estimator for SUSY
* ITHACA: ITHACA Tool for Handling Accelerated Cross-section Approximation (it's a [recursive acronym](https://en.wikipedia.org/wiki/Recursive_acronym) and of course the final destination in Homer's *Odyssey*)
  * Alternatively: Integrated Tool for Handling Accelerated Cross-section Approximation
* TAXES: Tool Accelerating X-section Estimation for SUSY (but no one would like us)
* PEXA: Program for Estimating X-sections Accurately 

##### 2. Project objective

Fast cross-section estimation at higher orders. *Fast* means: $\mathcal{O}(1\mathrm{\ s})$ evaluation time for 1 parameter point. More than 10 s would give trouble. Most points are thrown anyway.

Example: one point, with 10 experts and 36 processes, takes about 16 seconds right now.

Important issues include:

* Large amount of training data required (computationally expensive)
* Training time
* Generated file sizes (should be kept a lot smaller than a gigabyte)
* Obtained accuracy of the estimates
* Knowledge of the uncertainties
* Including all processes required (for strong LHC production): $\tilde g \tilde g, \tilde g \tilde q, \tilde q^\ast \tilde q, \tilde q \tilde q$

##### 3. Current status

***Boosted Decision Trees*** (Jon Vegard)

* Fast: due to the tree structure, not a lot of code is effectively executed
* Lots of training data required
* Easier to trim file sizes
* Tested on $\tilde g \tilde g$ production

***(Distributed) Gaussian Processes*** (Ingrid)

* Slow: evaluation time scales as $N^2$
* Not a lot of training data required
* Large files, harder to trim
* Tested on $\tilde q \tilde q$ production (expectation: $\tilde g \tilde g$ more involved for DGPs since more parameters)

***Latest developments***

* Set-up for semi-automated MSSM-24 sample generation with Prospino 2.1 (Jon Vegard)
  * On Abel: `/work/projects/nn9284k/xsec`
  * On GitHub: [jonvegards/prospino-PointSampler](https://github.com/jonvegards/prospino-PointSampler)
    * Point sampler calculates NLO cross-sections for strong and weak processes in MSSM-24 at four COM energies with scale variation

##### 4. Future work

* Including PDF uncertainties
  * Will require modifying Prospino with `COMMON` blocks
  * Switch to NNPDF
* Extend Gaussian Processes to all processes
  * Note: Electroweak production processes would take very long and require 4 times as many training points. Currently, we have 100 000 points for strong production
* Writing the actual program
  * C++ or Python?
    * Right now, bottle necks are the file size and evaluation time
    * It's good that the evaluation across experts (even across points) is parallelisable
    * In any case, it should be able to interface to GAMBIT. Is GAMBIT thread-safe? 
    * Probably best to have an unparallelised version first, going as fast as possible on one single thread, in order to avoid interfering with GAMBIT's two-level parallelisation
