Homogenized Inflatables
=======================

This is Part I of the codebase for our SIGGRAPH paper,
[Computational Homogenization for Inverse Design of Surface-based Inflatables](https://dl.acm.org/doi/10.1145/3658125).
The code is written primarily in C++, but it is meant to be used through the Python
bindings.

# Getting Started

## C++ Code Dependencies
The C++ code relies on `Boost`, which must be installed
separately.

The numerical solver depends on `Catamari`, which will be downloaded through the `cmake` file but require the `meson` build system. Alternatively one can choose to use the slower `CHOLMOD/UMFPACK` library, which must be installed
separately.

The code also relies on several dependencies that are included as submodules:
[MeshFEM](https://github.com/MeshFEM/MeshFEM),
[libigl](https://github.com/libigl/libigl),

Finally, it includes a version of Keenan Crane's [stripe patterns code](https://www.cs.cmu.edu/~kmcrane/Projects/StripePatterns/)
modified to generate fusing curve patterns and fix a few issues with boundary handling.

### macOS
You can install all the mandatory dependencies on macOS with [MacPorts](https://www.macports.org). When installing SuiteSparse, be sure to get a version linked against `Accelerate.framework` rather than `OpenBLAS`; on MacPorts this is achieved by requesting the `accelerate` variant, which is no longer the default. Simulations will run over 2x slower under `OpenBLAS`.

```bash
# Build/version control tools, C++ code dependencies
sudo port install cmake boost ninja meson cgal5
sudo port install SuiteSparse +accelerate
# Dependencies for jupyterlab/notebooks
sudo port install python39
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 17 && nvm use 17
# Dependencies for `shapely` module
sudo port install geos
```

### Ubuntu 20.04
A few more packages need to be installed on a fresh Ubuntu 20.04 install:
```bash
# Build/version control tools
sudo apt install git cmake ninja-build meson
# Dependencies for C++ code
sudo apt install libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libsuitesparse-dev
# Dependencies (pybind11, jupyterlab/notebooks)
sudo apt install python3-pip npm
sudo npm install npm@latest -g
# Dependencies for `shapely` module
sudo apt install libgeos-dev
```

## Obtaining and Building

Clone this repository *recursively* so that its submodules are also downloaded:

```bash
git clone --recursive https://github.com/jpanetta/Inflatables
```

Build the C++ code and its Python bindings using `cmake` and your favorite
build system. For example, with [`ninja`](https://ninja-build.org):

```bash
cd Inflatables
mkdir build && cd build
cmake .. -GNinja
ninja
```
Note that Catamari's performance is affected by the build settings, so to get best performance you'll want to choose the `Release` (not `RelWithAssert`) build type and enable `MESHFEM_VECTORIZE`. 

## Running the Jupyter Notebooks
The preferred way to interact with the inflatables code is in a Jupyter notebook,
using the Python bindings.
We recommend that you install the Python dependencies and JupyterLab itself in a
virtual environment (e.g., with [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [venv](https://docs.python.org/3/library/venv.html)).

```bash
pip3 install wheel # Needed if installing in a virtual environment
# Recent versions of jupyterlab and related packages cause problems:
#   JupyerLab 3.4 and later has a bug where the tab and status bar GUI
#                 remains visible after taking a viewer fullscreen
#   ipykernel > 5.5.5 clutters the notebook with stdout content
#   ipywidgets 8 and juptyerlab-widgets 3.0 break pythreejs
pip3 install jupyterlab==3.3.4 ipykernel==5.5.5 ipywidgets==7.7.2 jupyterlab-widgets==1.1.1
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .

pip3 install matplotlib scipy networkx libigl setproctitle gmsh multiprocess
pip3 install shapely # dependency of the fabrication file generation
```

Launch JupyterLab from the root python directory:
```bash
cd python
jupyter lab
```

Now try opening and running an demo notebook, e.g.,
[`python/unit_cell_experiments/cosine_curve.ipynb`](https://github.com/InflatableStructures/HomogenizedInflatables/blob/main/python/unit_cell_experiments/cosine_curve.ipynb).
