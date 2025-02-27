import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging

class SVDProcessor:
    def __init__(self, filepath: str, numerical_columns: list):
        self.filepath = filepath
        self.numerical_columns = numerical_columns
        self.data = None
        self.X_standardized = None
        self.U = None
        self.S = None
        self.Vt = None
        self.X_mean = None
        self.X_std = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess(self) -> None:
        try:
            self.data = pd.read_csv(self.filepath)
            for col in self.numerical_columns:
                if col in ['Selling Price', 'MRP']:
                    self.data[col] = self.data[col].replace('[,]', '', regex=True).astype(float)
            X = self.data[self.numerical_columns].values
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_standardized = (X - self.X_mean) / self.X_std
            self.logger.info("Data loaded and preprocessed successfully")
        except Exception as e:
            self.logger.error(f"Error in data loading/preprocessing: {str(e)}")
            raise

    def perform_svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            self.U, self.S, self.Vt = svd(self.X_standardized, full_matrices=False)
            self.logger.info("SVD decomposition completed")
            return self.U, self.S, self.Vt
        except Exception as e:
            self.logger.error(f"Error in SVD computation: {str(e)}")
            raise

    def get_explained_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        explained_variance_ratio = (self.S**2) / np.sum(self.S**2)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        return explained_variance_ratio, cumulative_variance_ratio

    def plot_variance(self, save_path: Optional[str] = None) -> None:
        try:
            _, cumulative_variance_ratio = self.get_explained_variance()
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                    cumulative_variance_ratio, 'bo-', linewidth=2)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('Explained Variance Ratio vs Number of Components')
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()
            self.logger.info("Variance plot generated")
        except Exception as e:
            self.logger.error(f"Error in plotting variance: {str(e)}")
            raise

    def reconstruct_data(self, k: int) -> np.ndarray:
        try:
            if self.U is None or self.S is None or self.Vt is None:
                raise ValueError("SVD not performed yet")
            reconstructed = np.dot(self.U[:, :k] * self.S[:k], self.Vt[:k, :])
            return (reconstructed * self.X_std) + self.X_mean
        except Exception as e:
            self.logger.error(f"Error in data reconstruction: {str(e)}")
            raise

    def calculate_reconstruction_error(self, k: int) -> float:
        reconstructed = self.reconstruct_data(k)
        standardized_reconstructed = (reconstructed - self.X_mean) / self.X_std
        return float(np.mean((self.X_standardized - standardized_reconstructed)**2))

    def process_and_analyze(self, k: int = 2) -> pd.DataFrame:
        try:
            self.load_and_preprocess()
            self.perform_svd()
            explained_var, cumulative_var = self.get_explained_variance()
            reconstructed_data = self.reconstruct_data(k)
            error = self.calculate_reconstruction_error(k)
            reconstructed_cols = [f'{col}_reconstructed' for col in self.numerical_columns]
            result_df = pd.concat([
                self.data,
                pd.DataFrame(reconstructed_data, columns=reconstructed_cols)
            ], axis=1)
            self.logger.info(f"Singular Values: {self.S}")
            self.logger.info(f"Explained Variance Ratio: {explained_var}")
            self.logger.info(f"Cumulative Variance Ratio: {cumulative_var}")
            self.logger.info(f"Reconstruction Error with {k} components: {error}")
            return result_df
        except Exception as e:
            self.logger.error(f"Error in process_and_analyze: {str(e)}")
            raise

def main():
    numerical_columns = ['Ratings', 'no_ratings', 'Selling Price', 'MRP', 'Discount']
    processor = SVDProcessor('products.csv', numerical_columns)
    result_df = processor.process_and_analyze(k=2)
    processor.plot_variance()
    display_columns = numerical_columns + [f'{col}_reconstructed' for col in numerical_columns]
    print("\nOriginal vs Reconstructed Data (first 5 rows):")
    print(result_df[display_columns].head())

if __name__ == "__main__":
    main()
