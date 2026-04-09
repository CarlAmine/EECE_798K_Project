"""
SINDy model: sequential thresholded least squares sparse identification.
"""

import numpy as np
from scipy.linalg import lstsq
import warnings


class SINDyModel:
    """
    Sequential Thresholded Least Squares SINDy model.
    
    Fits sparse models: dx_i/dt = library @ coefficients_i
    """
    
    def __init__(self, threshold: float = 0.1, max_iter: int = 10,
                 convergence_tol: float = 1e-5):
        """
        Initialize SINDy model.
        
        Parameters
        ----------
        threshold : float
            Threshold for coefficient magnitude (sparsification)
        max_iter : int
            Maximum iterations for sequential thresholding
        convergence_tol : float
            Convergence tolerance
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        
        self.coefficients = None
        self.residuals = None
        self.library_descriptions = None
    
    def fit(self, X: np.ndarray, Xdot: np.ndarray, 
            library: np.ndarray,
            library_descriptions: list = None) -> dict:
        """
        Fit SINDy model using sequential thresholded least squares.
        
        Parameters
        ----------
        X : np.ndarray
            State data (n_samples, n_vars)
        Xdot : np.ndarray
            State derivatives (n_samples, n_vars)
        library : np.ndarray
            Candidate library (n_samples, n_terms)
        library_descriptions : list, optional
            Descriptions of library terms
            
        Returns
        -------
        dict
            Fit diagnostics
        """
        X = np.atleast_2d(X)
        Xdot = np.atleast_2d(Xdot)
        library = np.atleast_2d(library)
        
        # Ensure correct shapes
        if X.shape[0] == 1:
            X = X.T
        if Xdot.shape[0] == 1:
            Xdot = Xdot.T
        if library.shape[0] == 1:
            library = library.T
        
        n_samples, n_vars = X.shape
        n_terms = library.shape[1]
        
        self.library_descriptions = library_descriptions or [f't{i}' for i in range(n_terms)]
        
        # Fit each variable independently
        self.coefficients = np.zeros((n_terms, n_vars))
        self.residuals = np.zeros(n_vars)
        sparse_counts = np.zeros(n_vars, dtype=int)
        
        for i in range(n_vars):
            y = Xdot[:, i]
            coeff = self._fit_sequential_threshold(library, y)
            self.coefficients[:, i] = coeff
            
            # Compute residual
            y_pred = library @ coeff
            self.residuals[i] = np.linalg.norm(y - y_pred) / np.linalg.norm(y)
            sparse_counts[i] = np.sum(np.abs(coeff) > 0)
        
        diagnostics = {
            'residuals': self.residuals,
            'sparse_terms_per_var': sparse_counts,
            'total_sparse_terms': np.sum(sparse_counts),
        }
        
        return diagnostics
    
    def _fit_sequential_threshold(self, library: np.ndarray,
                                  y: np.ndarray) -> np.ndarray:
        """
        Fit using sequential thresholded least squares for a single target.
        
        Parameters
        ----------
        library : np.ndarray
            Candidate library (n_samples, n_terms)
        y : np.ndarray
            Target derivative (n_samples,)
            
        Returns
        -------
        np.ndarray
            Sparse coefficient vector
        """
        lib_active = library.copy()
        coeff = np.zeros(library.shape[1])
        
        for iteration in range(self.max_iter):
            # Least squares fit
            if lib_active.shape[1] == 0:
                break
            
            coeff_active, _, _, _ = lstsq(lib_active, y)
            
            # Zero out small coefficients
            small_idx = np.abs(coeff_active) < self.threshold
            
            if np.sum(small_idx) == 0:
                # Convergence: no new zeros
                if iteration > 0:
                    break
            
            # Remove small terms from active library
            lib_active = lib_active[:, ~small_idx]
            
            # Map back to original indices
            active_mask = np.ones(library.shape[1], dtype=bool)
            for j in range(library.shape[1]):
                if j >= len(active_mask) - np.sum(small_idx):
                    active_mask[j] = False
            
        # Final fit with remaining active terms
        if lib_active.shape[1] > 0:
            coeff_final, _, _, _ = lstsq(lib_active, y)
            
            # Fill in final coefficients
            active_indices = np.nonzero(~small_idx)[0]
            coeff_final_full = np.zeros(library.shape[1])
            coeff_final_full[active_indices] = coeff_final
            coeff = coeff_final_full
        
        return coeff
    
    def predict(self, X: np.ndarray, library: np.ndarray) -> np.ndarray:
        """
        Predict derivatives given state and library.
        
        Parameters
        ----------
        X : np.ndarray
            State data (n_samples, n_vars)
        library : np.ndarray
            Evaluated library (n_samples, n_terms)
            
        Returns
        -------
        np.ndarray
            Predicted derivatives (n_samples, n_vars)
        """
        if self.coefficients is None:
            raise RuntimeError("Model not fitted yet")
        
        library = np.atleast_2d(library)
        if library.shape[0] == 1:
            library = library.T
        
        Xdot_pred = library @ self.coefficients
        return Xdot_pred
    
    def print_equations(self, var_names: list = None):
        """
        Print discovered equations in readable form.
        
        Parameters
        ----------
        var_names : list, optional
            Names of state variables
        """
        if self.coefficients is None:
            print("Model not fitted yet")
            return
        
        var_names = var_names or [f'x{i}' for i in range(self.coefficients.shape[1])]
        
        print("\nDiscovered Equations:")
        print("=" * 60)
        
        for i, var_name in enumerate(var_names):
            coeff = self.coefficients[:, i]
            nonzero_idx = np.nonzero(coeff)[0]
            
            if len(nonzero_idx) == 0:
                print(f"d{var_name}/dt = 0")
                continue
            
            terms = []
            for j in nonzero_idx:
                c = coeff[j]
                term_name = self.library_descriptions[j] if self.library_descriptions else f't{j}'
                
                if c > 0:
                    sign = '+' if terms else ''
                else:
                    sign = '-'
                    c = abs(c)
                
                if abs(c - 1.0) < 1e-6:
                    terms.append(f"{sign} {term_name}")
                else:
                    terms.append(f"{sign} {c:.4f}*{term_name}")
            
            eq = f"d{var_name}/dt = " + " ".join(terms).lstrip('+ ')
            print(eq)
        
        print("=" * 60)
        if self.residuals is not None:
            print(f"Relative errors: {self.residuals}")


class SINDy:
    """Simplified SINDy wrapper combining model, library, and fitting."""
    
    def __init__(self, threshold: float = 0.1, max_iter: int = 10):
        self.model = SINDyModel(threshold=threshold, max_iter=max_iter)
    
    def fit_polynomial(self, X: np.ndarray, Xdot: np.ndarray,
                      max_degree: int = 2, var_names: list = None) -> dict:
        """Fit with polynomial library."""
        from library import build_library_polynomial
        
        library, descriptions = build_library_polynomial(
            X, max_degree, var_names
        )
        
        return self.model.fit(X, Xdot, library, descriptions)
    
    def fit_physics(self, X: np.ndarray, Xdot: np.ndarray,
                   var_names: list = None) -> dict:
        """Fit with physics-guided library."""
        from library import build_library_physics
        
        library, descriptions = build_library_physics(X, var_names)
        
        return self.model.fit(X, Xdot, library, descriptions)
    
    def predict(self, X: np.ndarray, library_fn) -> np.ndarray:
        """Predict using fitted model."""
        library = library_fn(X)
        return self.model.predict(X, library)
