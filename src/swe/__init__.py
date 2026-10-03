"""Displacement shallow-water engine. Graphics and CUDA are optional extras."""
from .mesh import Mesh
from .terrain import FlatBottom, GaussianBottom, ParabolicBottom, GridBottom
from .core import ModelBuilder, Model, State, DisplacementSolver, NumericalError, load_checkpoint
from .runtime import run

__version__="0.1.0"
__all__=["Mesh","FlatBottom","GaussianBottom","ParabolicBottom","GridBottom",
         "ModelBuilder","Model","State","DisplacementSolver","NumericalError","load_checkpoint","run"]
