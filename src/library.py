"""
SINDy candidate library generation.
Build polynomial and nonlinear function libraries for sparse identification.
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement


class LibraryBuilder:
    """Build candidate function libraries for SINDy."""
    
    def __init__(self, n_vars: int = None, var_names: list = None):
        """
        Initialize library builder.
        
        Parameters
        ----------
        n_vars : int, optional
            Number of variables
        var_names : list, optional
            Names of variables (for display)
        """
        self.n_vars = n_vars
        self.var_names = var_names or [f'x{i}' for i in range(n_vars)] if n_vars else []
        self.library = None
        self.terms = []
        self.descriptions = []
    
    def build_polynomial_library(self, X: np.ndarray, max_degree: int = 2) -> np.ndarray:
        """
        Build polynomial library.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_vars)
        max_degree : int
            Maximum polynomial degree
            
        Returns
        -------
        np.ndarray
            Library matrix (n_samples, n_terms)
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T  # Transpose if needed
        
        n_samples, n_vars = X.shape
        self.n_vars = n_vars
        
        library_list = []
        self.terms = []
        self.descriptions = []
        
        # Constant term
        library_list.append(np.ones(n_samples))
        self.terms.append([0] * n_vars)
        self.descriptions.append('1')
        
        if max_degree >= 1:
            # Linear terms
            for i in range(n_vars):
                library_list.append(X[:, i])
                powers = [0] * n_vars
                powers[i] = 1
                self.terms.append(powers)
                self.descriptions.append(self.var_names[i] if self.var_names else f'x{i}')
        
        if max_degree >= 2:
            # Quadratic terms
            for i in range(n_vars):
                for j in range(i, n_vars):
                    term_data = X[:, i] * X[:, j]
                    library_list.append(term_data)
                    powers = [0] * n_vars
                    powers[i] += 1
                    powers[j] += 1
                    self.terms.append(powers)
                    
                    if i == j:
                        desc = f'{self.var_names[i]}^2' if self.var_names else f'x{i}^2'
                    else:
                        desc = f'{self.var_names[i]}*{self.var_names[j]}' if self.var_names else f'x{i}*x{j}'
                    self.descriptions.append(desc)
        
        if max_degree >= 3:
            # Cubic terms
            for i in range(n_vars):
                for j in range(i, n_vars):
                    for k in range(j, n_vars):
                        term_data = X[:, i] * X[:, j] * X[:, k]
                        library_list.append(term_data)
                        powers = [0] * n_vars
                        powers[i] += 1
                        powers[j] += 1
                        powers[k] += 1
                        self.terms.append(powers)
                        
                        if i == j == k:
                            desc = f'{self.var_names[i]}^3' if self.var_names else f'x{i}^3'
                        elif i == j:
                            desc = f'{self.var_names[i]}^2*{self.var_names[k]}' if self.var_names else f'x{i}^2*x{k}'
                        else:
                            desc = f'{self.var_names[i]}*{self.var_names[j]}*{self.var_names[k]}' if self.var_names else f'x{i}*x{j}*x{k}'
                        self.descriptions.append(desc)
        
        self.library = np.column_stack(library_list)
        return self.library
    
    def build_physics_library(self, X: np.ndarray) -> np.ndarray:
        """
        Build a physics-guided library for stick-slip dynamics.
        Includes: constant, linear, quadratic, and log-velocity terms.
        
        Parameters
        ----------
        X : np.ndarray
            Data (n_samples, n_vars) with columns [shear_stress, slip_velocity]
            or [acceleration_z, velocity_z]
            
        Returns
        -------
        np.ndarray
            Library matrix
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        
        n_samples = X.shape[0]
        library_list = []
        self.terms = []
        self.descriptions = []
        
        # Terms:
        # 1 (constant/static)
        library_list.append(np.ones(n_samples))
        self.descriptions.append('1')
        
        # x (first variable)
        library_list.append(X[:, 0])
        self.descriptions.append('x[0]')
        
        # v (second variable)
        library_list.append(X[:, 1])
        self.descriptions.append('x[1]')
        
        # x^2
        library_list.append(X[:, 0] ** 2)
        self.descriptions.append('x[0]^2')
        
        # x*v
        library_list.append(X[:, 0] * X[:, 1])
        self.descriptions.append('x[0]*x[1]')
        
        # v^2
        library_list.append(X[:, 1] ** 2)
        self.descriptions.append('x[1]^2')
        
        # log(|v| + eps) - for friction-like behavior
        eps = 1e-6
        log_v = np.log(np.abs(X[:, 1]) + eps)
        library_list.append(log_v)
        self.descriptions.append('log(|x[1]|+eps)')
        
        self.library = np.column_stack(library_list)
        return self.library
    
    def get_descriptions(self) -> list:
        """Get human-readable descriptions of library terms."""
        return self.descriptions
    
    def print_library_info(self):
        """Print information about the current library."""
        if self.library is None:
            print("Library not built yet.")
            return
        
        print(f"Library size: {self.library.shape[0]} samples x {self.library.shape[1]} terms")
        print(f"Terms:")
        for i, desc in enumerate(self.descriptions):
            print(f"  [{i}] {desc}")


def build_library_polynomial(X: np.ndarray, max_degree: int = 2,
                            var_names: list = None) -> tuple:
    """
    Convenience function to build polynomial library.
    
    Parameters
    ----------
    X : np.ndarray
        Data (n_samples, n_vars)
    max_degree : int
        Maximum polynomial degree
    var_names : list, optional
        Variable names for descriptions
        
    Returns
    -------
    tuple
        (library_matrix, descriptions)
    """
    builder = LibraryBuilder(X.shape[1], var_names)
    lib = builder.build_polynomial_library(X, max_degree)
    return lib, builder.get_descriptions()


def build_library_physics(X: np.ndarray, var_names: list = None) -> tuple:
    """
    Convenience function to build physics-guided library.
    
    Parameters
    ----------
    X : np.ndarray
        Data (n_samples, 2) - [stress/accel, velocity]
    var_names : list, optional
        Variable names
        
    Returns
    -------
    tuple
        (library_matrix, descriptions)
    """
    builder = LibraryBuilder(var_names=var_names)
    lib = builder.build_physics_library(X)
    return lib, builder.get_descriptions()
