from pathlib import Path # To handle file paths 
from typing import Dict, List, Tuple, Optional 

import numpy as np #For Numerical operations and arrays 
import pandas as pd #For data manipulation (DataFrames)
import matplotlib.pyplot as plt #For creating plots 
#sklearn imports for machine learning 
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler #Normalize data
from sklearn.impute import SimpleImputer #Fill missing Values
from sklearn.compose import ColumnTransformer #Apply transfroms to specific columns
from sklearn.pipeline import Pipeline #Chain preprocessing + model together
# Metric for Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
#Different classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# -------------------------------------------------------------------
# CONFIGURATION / PATHS
# -------------------------------------------------------------------


DATASET_1_PATH = "dataset_1.csv" #Takes data set 1 from the folder
DATASET_2_PATH = "dataset_2.csv" #Takes data set 2 from the folder

PLOTS_ROOT = Path("plots")
PLOTS_ROOT.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 1. PREPROCESSOR CLASS
# -------------------------------------------------------------------

class Preprocessor:
    """
    Handles:
      - loading data from CSV
      - basic inspection / EDA-style info
      - splitting into train / test sets
      - constructing a preprocessing pipeline
        (imputation + standardisation)
    """

    def __init__(self, data_path: str, target_col: str):
        """
        Constructor- runs when you create a Preprocessor object
        Args:
            data_path: Path to the CSV file (e.g "dataset_1.csv")
            target_col: Name of the column we're trying to predict (e.g., "label")
        EXAMPLE USAGE:
            Pre= Preprocessor("dataset_1.csv", target_col="label")
        """
        self.data_path = Path(data_path) #convert to path object
        self.target_col = target_col #save the target column name 
        self.df: Optional[pd.DataFrame] = None #DataFrame placeholder; set once data is loaded

    # ---- data loading / info -------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame

        Returns:
            The loaded DataFrame
        """
        print(f"\n[Preprocessor] Loading data from {self.data_path} ...")
        try:
            self.df = pd.read_csv(self.data_path) #Reads CSV
        except FileNotFoundError as e:
            #Custom message, then re-raise so the caller knows it
            raise FileNotFoundError(f"ERROR: File '{self.data_path}' not found.") from e
        print(f"[Preprocessor] Shape: {self.df.shape}")#Shows dimensions
        return self.df

    def show_basic_info(self) -> None:
        """
        Display basic information about the dataset
        
        shows:
        -First few rows (.head())
        -Statistical summary(.describe())
        -Missing value counts
        -Target variable distribution
        """
        if self.df is None: #For safety check
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n[Preprocessor] Head:")#First 5 rows
        print(self.df.head())
        print("\n[Preprocessor] Describe:")#Statistics
        print(self.df.describe(include="all"))
        print("\n[Preprocessor] Missing values per column:")
        print(self.df.isna().sum())
        print("\n[Preprocessor] Target value counts:")
        print(self.df[self.target_col].value_counts())

    # ---- splitting -----------------------------------------------------------
    # These helper methods split the dataset into:
    # x : the input features ( everything except the target column )
    # y : the target labels ( the column we want to predict )

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """ Seperate the loaded DataFrame into features (x) and target (y) 
        
        Returns
        -------
        x : pd.DataFrame
            All columns except the target column 
        y : pd.Series
            The target column that we want the model to predict
        Raises
        -------
        ValueError
            If no data has been loaded into 'self. df' yet.
        """
        # Makes sure the data has been loaded before trying to split it.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Drop the target column to obtain the feature matrix x.
        X = self.df.drop(columns=[self.target_col])
        # Extract only the target column as a 1D series y.
        y = self.df[self.target_col]
        return X, y

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split features and target into training and test sets.
        Parameters
        -------
        test_size : float, default =0.2
            Proportion of the dataset to allocate to the test set
            (e.g. 0.2 = 20% test,80% train).
        random_state : int , default =42
            seed for the random number generator so the split is reproducible.
        stratify : bool, default =True 
            If True, the class distribution of y is preserved in both 
            the train and test sets ( stratified split ).
            Returns
            -------
            x_train, x_test : pd.DataFrame
                Feature matrices for training and testing.
            y_train, y_test : pd.Series
                Target vectors for training and testing.
                """
        # First obtains X (features) and y (target) from the full DataFrame.
        X, y = self.get_features_and_target()
        # Decides whether to use stratification: preserve label proportions if requested. 
        y_strat = y if stratify else None
        # Use sklearn's train_test_split to perform the actual splitting.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y_strat
        )
        # Print the shapes so one can quickly check the split worked as expected.
        print(
            f"\n[Preprocessor] Train shape: {X_train.shape}, "
            f"Test shape: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    # ---- preprocessing pipeline ---------------------------------------------

    def build_numeric_pipeline(
        self,
        selected_features: Optional[List[str]] = None,
    ) -> Tuple[ColumnTransformer, List[str]]:
        """
        Build a preprocessing pipeline for the numeric features.

    The returned ColumnTransformer:
      - fills in missing values using the median of each feature
      - standardises features to have mean 0 and standard deviation 1

    Parameters
    ----------
    selected_features : list of str, optional
        Names of the features to include in the pipeline.
        If None, all non-target columns in the DataFrame are used.

    Returns
    -------
    preprocessor : ColumnTransformer
        A transformer that applies the numeric preprocessing steps to
        the chosen feature columns.
    feature_names : list of str
        The list of feature names actually used in the pipeline.
        """
        # Ensures that data has been loaded before building a pipeline.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Decides which feature to use: 
        # - If no subset is provide, uses all columns except the target.
        # -otherwise, uses only the user-specified subset.

        if selected_features is None:
            feature_names = self.df.drop(columns=[self.target_col]).columns.tolist()
        else:
            feature_names = selected_features

        #Define a small sklearn Pipeline for numeric features :
        # 1) SimpleImputer replaces missing values with the median.
        # 2) StandardScaler standardies each feature (mean=0, std=1)
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        # Wraps the numeric transformer in a columnTransformer so that
        # It is applied only to the choosen feature columns.

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, feature_names),
            ]
        )
        # Print which features are being used, for quick debugging
        print(f"[Preprocessor] Using features: {feature_names}")
        return preprocessor, feature_names


# -------------------------------------------------------------------
# 2. CLASSIFIER (used as BinaryClassifier for dataset 1)
# -------------------------------------------------------------------

class Classifier:
    """
     Generic classifier wrapper which bundles together:
      - a scikit-learn model (e.g. LogisticRegression, RandomForest, KNN, SVM)
      - the preprocessing pipeline (e.g. imputation + scaling)

    This class exposes a simple API:
      - .fit(X_train, y_train)
      - .predict(X_test)
      - .get_feature_importances()
    """

    def __init__(
        self,
        model_name: str,
        preprocessor: ColumnTransformer,
        feature_names: List[str],
    ):
        """
        Initialise the classifier wrapper.

        Parameters
        ----------
        model_name : str
            Short name indicating which classifier to use.
            Supported values: "logistic", "random_forest",
            "decision_tree", "knn_5", "svm".
        preprocessor : ColumnTransformer
            Preprocessing pipeline that will be applied to the input
            features before the classifier (e.g. imputer + scaler).
        feature_names : list of str
            Names of the features used by the model. These are needed
            later when computing and displaying feature importances.
            """
        #Storesd basic configuration for later use.
        self.model_name = model_name
        self.feature_names = feature_names
        # ------------------------------------------------------------------
        # Map the string `model_name` to an actual scikit-learn model.
        # This keeps the rest of the code clean: can just pass a name like
        # "random_forest" instead of manually constructing each model.
        # ------------------------------------------------------------------

        
        if model_name == "logistic":
            # Logistic Regression for linear decision boundries.
            model = LogisticRegression(max_iter=1000)
        elif model_name == "random_forest":
            # Random Forest with 200 trees; fixed random_state for reproducibility.
            model = RandomForestClassifier(
                n_estimators=200, random_state=42
            )
        elif model_name == "decision_tree":
            # Single Decision Tree classifier.
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "knn_5":
            # k-Nearest Neighbours with k =5.
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "svm":
            # SVM Vector Machine using an RBF ( Gaussian) kernel.
            model = SVC(kernel="rbf", gamma="scale")
        else:
            # Defensive programming: catches typos or unsupported model names
            raise ValueError(f"Unknown model name: {model_name}")
        
        # ------------------------------------------------------------------
        # Combine the preprocessing steps and the classifier into a single
        # scikit-learn Pipeline. When we call .fit() or .predict() on this
        # pipeline, it will:
        #   1) apply the preprocessor to X
        #   2) then feed the transformed data into the chosen model.
        # This guarantees that training and inference use exactly the same
        # preprocessing steps.
        # ------------------------------------------------------------------

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

    # ---- basic API -----------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Classifier":
        """
        Fit the underlying pipeline (preprocessor + classifier) on the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.

    Returns
    -------
    self : Classifier
        Returns the instance so that calls can be chained if desired.
    """
        # Logs which model is being trained.
        print(f"\n[Classifier] Fitting model: {self.model_name}")
        # Fits the full pipeline: first preprocessor, then classifier.
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """ 
        Generate predictions for the given feature matrix.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels for each row in X_test.
        """
        # The pipeline will automatically apply the same preprocessing.
        # Steps are used during training before making predictions.
        return self.pipeline.predict(X_test)

    # ---- feature importance --------------------------------------------------

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Extract feature importances from the underlying classifier, if available.

    This method tries to provide a unified interface for different model types:

    - For linear models (e.g. LogisticRegression), it uses the absolute
      value of the coefficients stored in `coef_`.
    - For tree-based models (e.g. RandomForest), it uses `feature_importances_`.
    - For models that do not expose feature importances (e.g. KNN, RBF SVM),
      the method returns None.

    Returns
    -------
    importance_dict : dict[str, float] or None
        A dictionary mapping feature names to importance values, sorted
        from most important to least important. Returns None if the
        classifier does not provide any notion of feature importance.
        """
        # Get a handle to the actual classifier object inside the pipeline.
        clf = self.pipeline.named_steps["classifier"]
        
        # ---------- Determine raw importance values depending on model type -----

        # Case 1: linear-type models that expose 'coef_' (e.g LogisticRegression)
        if hasattr(clf, "coef_"):
            # Flatten the coefficient array and take absolute values so that 
            # negative and positive coefficients are treated symmetrically.
            coef = np.ravel(clf.coef_)
            importances = np.abs(coef)
            print("[Classifier] Using absolute coefficients as importance.")
        # Case 2 : tree-based models with 'feature_importances_' attribute
        elif hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            print("[Classifier] Using feature_importances_ from tree model.")
        # Case 3 : models that do not expose feature importances.
        else:
            print(
                "[Classifier] Model does not expose feature importances "
                f"({self.model_name})."
            )
            return None
        # ---------- Map importance values back to feature names -----------------

        # Zips feature names with their importance values into a dictionary.
        importance_dict = dict(zip(self.feature_names, importances))
        # Sorts the dictionary by importance (descending order) so that 
        # the most influential feature appears first.
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
        return importance_dict


# For Dataset 1 we can simply use Classifier as a BinaryClassifier:
BinaryClassifier = Classifier


# -------------------------------------------------------------------
# 3. EVALUATOR CLASS
# -------------------------------------------------------------------

class Evaluator:
    """
    Utility class for evaluating models and creating visualisations.

    It is responsible for:
      - computing standard classification metrics
      - plotting confusion matrices
      - plotting feature importance charts
      - plotting accuracy vs number-of-features curves (Dataset 1)
      - plotting learning curves (Dataset 2)
    """

    def __init__(self, output_dir: Path):
        """Initialise the evaluator.

        Parameters
        ----------
        output_dir : Path
            Directory where all plots and figures will be saved.
        """
        # Stores the directory path.
        self.output_dir = output_dir
        # Ensures that the directory (and any mssing parent folders)
        # actually exists before we start saving plots.
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- metrics -------------------------------------------------------------

    def metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """Compute and print standard classification metrics.

        Parameters
        ----------
        y_true : pd.Series
            Ground-truth labels from the dataset.
        y_pred : np.ndarray
            Labels predicted by the model.
        model_name : str
            Name of the model (used only for printing / logging).

        Returns
        -------
        metrics_dict : dict[str, float]
            Dictionary containing accuracy, precision, recall and F1 score.
            """
        # Overall proportion of correct predictions.
        acc = accuracy_score(y_true, y_pred)

        # Precision, recall and F1 scores are averaged across classes.
        # 'weighted' ensures each class is weighted by its support (size),
        # and zero_division=0 avoids errors if a class is never predicted.
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        # Nicely formatted summary printed to the console
        print(f"\n[Evaluator] Metrics for {model_name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Returns the scores in a convenient dictionary form.
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ---- confusion matrix ----------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        filename: str,
    ) -> None:
        """
        Plot and save the confusion matrix for a set of predictions.

        Parameters
        ----------
        y_true : pd.Series
            Ground-truth class labels.
        y_pred : np.ndarray
            Predicted class labels produced by the model.
        model_name : str
            Name of the model; used in the plot title.
        filename : str
            Name of the file (e.g. "cm_logistic.png") to save the figure as
            inside the evaluator's output directory.
            """
        # Gets the sorted list of unique class labels so that the
        # confusion matrix axes are ordered consistently.
        labels = sorted(pd.unique(y_true))

        # Computes the raw confusion matrix counts.
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Creates a new figure and axes for the plot.
        fig, ax = plt.subplots(figsize=(5, 5))

        # Wraps the confusion matrix in a ConfusionMatrixDisplay, which
        # knows how to draw a nicely formatted heatmap.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        # Actually draws the confusion matrix on the axes.
        # Blue colour map is used and suppressed the colourbar for simplicity.
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        # Adds a tittle indicating which model this confusion matrix belongs to.
        ax.set_title(f"Confusion matrix: {model_name}")

        # Adjusts layout so label and tittle are not cut off in the saved file.
        plt.tight_layout()

        # Builds the full output path and saves the figure as an image file.
        path = self.output_dir / filename
        plt.savefig(path)
        
        # Closes the figure to free memory and avoids overlapping plots later.
        plt.close()

        # Logs where the confusion matrix was saved.
        print(f"[Evaluator] Saved confusion matrix to {path}")

    # ---- feature importance --------------------------------------------------

    def plot_feature_importances(
        self,
        feature_importances: Dict[str, float],
        title: str,
        filename: str,
        top_n: Optional[int] = None,
    ) -> None:   
        """
    Plots feature importance values for Dataset 1. 
     Can optionally limit to the top N most important features.
     """ 
        if not feature_importances:
            print("[Evaluator] No feature importances to plot.")
            return
        items = list(feature_importances.items()) # if dictionary is empty, nothing to plot
        # Optionally, keep only the first N(already sorted before passed here) 
        if top_n is not None:
            items = items[:top_n]

        # Split into two lists: feature names and numeric importance values
        features, values = zip(*items)

        # Create figure and horizontal bar chart
        plt.figure(figsize=(8, 5))
        y_pos = np.arange(len(features)) # Numeric y-axis positions
        plt.barh(y_pos, values) # Horizontal bar plot
        plt.yticks(y_pos, features) # Label bars with feature names
        plt.gca().invert_yaxis() # Highest importance at the top
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()

        # Save to evaluator output directory 
        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved feature importance plot to {path}")

    # ---- feature subset performance (Dataset 1) ------------------------------

    def plot_feature_subset_performance(
        self,
        subset_sizes: List[int],
        accuracies: List[float],
        title: str,
        filename: str,
    ) -> None: # 
        """
    Plots accuracy as a function of the number of selected features. 
    This is used when evaluating different feature subsets for Dataset 1
    """
        plt.figure(figsize=(6, 4))
        plt.plot(subset_sizes, accuracies, marker="o") # Line with circular markers
        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.grid(True) # Add grid for readability
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved feature-subset performance plot to {path}")

    # ---- learning curve (Dataset 2) -----------------------------------------

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        title: str,
        filename: str,
    ) -> None:
        """
    Creates a learning curve to show how training/test accuracy changes as a function of training set size.
    Parameters:
    - train_sizes: Array of traing set sizes used for evaluation.
    - train_scores: 2D array of training scores for each CV fold and training size.
    - test_scores: 2D array of validation scores for each CV fold and training size.
    - title: Title of the plot.
    - filename: Name of the file to save the figure.
    It is used for dataset 2 to determine minimum required data for 70% accuracy, and also diagnoses underfitting/overfitting
    """ 
    # Compute mean and standard deviation across CV folds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(7, 5))

        # Plots the mean training score curve: Smaller markers and thinner lines so the dense curve is clearer
        plt.plot(
            train_sizes,
            train_mean,
            marker="o",
            linestyle="-",
            markersize=3,
            linewidth=1.0,
            label="Training score",
        ) # Plots the mean cross-validation curve
        plt.plot(
            train_sizes,
            test_mean,
            marker="o",
            linestyle="-",
            markersize=3,
            linewidth=1.0,
            label="Cross-validation score",
        )
        # Adds shaded regions representing +/-1 standard deviation 
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.15,
        )

        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        # Saves the plot to the specified path
        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"[Evaluator] Saved learning curve to {path}")

    # ---- CV accuracy bar chart (Dataset 2) ----------------------------------

    def plot_cv_accuracy_bar(
        # Bar chart for mean CV accuracy of each candidate model.
        # Y-axis is restricted to 0.90–1.01 so small differences
        # between 0.93 and 1.00 are clearly visible.
        self,
        model_names: List[str],
        cv_means: List[float],
        title: str,
        filename: str,
    ) -> None:
        """
        Bar chart of mean cross-validation accuracy for each model.
        Y-axis is zoomed so differences between ~0.93–1.00 are visible.
        """
        plt.figure(figsize=(7, 5)) # Set figure size 

        indices = np.arange(len(model_names)) # Numerical positions for bars
        plt.bar(indices, cv_means) # Draws the bars

        plt.xticks(indices, model_names, rotation=15) # Labels each bar with model name
        plt.ylabel("Mean CV accuracy") # Y - Axis Label
        plt.title(title) # Plot title

        # Zoom y-axis to highlight differences between top models
        plt.ylim(0.90, 1.00)

        plt.grid(axis="y", linestyle="--", alpha=0.4) # Adds light horizontal grid lines for easier comparison
        plt.tight_layout() # Adjusts spacing

        path = self.output_dir / filename # Outputs file path
        plt.savefig(path) # Saves the figure
        plt.close()
        print(f"[Evaluator] Saved CV accuracy bar chart to {path}")


# -------------------------------------------------------------------
# 4. PIPELINES FOR DATASET 1 AND DATASET 2
# -------------------------------------------------------------------

PLOTS_ROOT = Path("plots")
PLOTS_ROOT.mkdir(exist_ok=True)


def run_dataset1_pipeline():
    """
    High-level pipeline for Dataset 1:

    1. Load and inspect the materials dataset.
    2. Build a numeric preprocessing pipeline.
    3. Train Logistic Regression and Random Forest models.
    4. Compare their test accuracy and keep the best model.
    5. For the best model:
         - plot feature importances
         - rerun Logistic Regression with progressively fewer top features
           and plot accuracy vs number of features.
    """
    print("\n" + "=" * 70)
    print("DATASET 1: Binary classification and feature selection")
    print("=" * 70)

    # --- step 1: preprocessing / splitting -----------------------------------
    pre = Preprocessor(DATASET_1_PATH, target_col="label")
    pre.load_data()
    pre.show_basic_info()
    preprocessor_all, feature_names = pre.build_numeric_pipeline()
    X_train, X_test, y_train, y_test = pre.train_test_split(test_size=0.2, stratify=True)

    evaluator = Evaluator(PLOTS_ROOT / "dataset1")

    # --- step 2: try two binary classifiers ----------------------------------
    models = {
        "LogisticRegression": BinaryClassifier(
            "logistic", preprocessor_all, feature_names
        ),
        "RandomForest": BinaryClassifier(
            "random_forest", preprocessor_all, feature_names
        ),
    }

    best_name = None
    best_acc = -np.inf
    best_model: Optional[BinaryClassifier] = None

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = evaluator.metrics(y_test, y_pred, name)
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name=name, filename=f"confusion_{name}.png"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_model = clf

    print(
        f"\n[Dataset 1] Best test accuracy: {best_acc:.4f} "
        f"with model {best_name}"
    )

    # --- step 3: feature importance and subsets ------------------------------

    if best_model is None:
        return

    importances = best_model.get_feature_importances()
    if importances is None:
        return

    evaluator.plot_feature_importances(
        importances,
        title=f"Feature importances ({best_name})",
        filename="feature_importances.png",
    )

    # ranked_features is ordered by importance (most important first)
    ranked_features = list(importances.keys())
    n_features = len(ranked_features)

    # Evaluate ALL subset sizes: n_features, n_features-1, ..., 1
    subset_sizes = list(range(n_features, 0, -1))
    subset_accuracies = []

    print("\n[Dataset 1] Feature subset experiments (all sizes):")
    for k in subset_sizes:
        top_feats = ranked_features[:k]
        print(f"  Using top {k} features: {top_feats}")

        preproc_k, feats_k = pre.build_numeric_pipeline(selected_features=top_feats)
        clf_k = BinaryClassifier("logistic", preproc_k, feats_k)
        clf_k.fit(X_train, y_train)
        y_pred_k = clf_k.predict(X_test)
        metrics_k = evaluator.metrics(y_test, y_pred_k, f"LogReg_top_{k}")
        subset_accuracies.append(metrics_k["accuracy"])

    evaluator.plot_feature_subset_performance(
        subset_sizes,
        subset_accuracies,
        title="Accuracy vs number of features (LogReg)",
        filename="feature_subset_accuracy.png",
    )


def run_dataset2_pipeline():
    """
    High-level pipeline for Dataset 2:

    1. Load and inspect the binary dataset (labels 0/1).
    2. Build a numeric preprocessing pipeline.
    3. Define five candidate models:
         - Logistic Regression
         - KNN (k = 5)
         - Random Forest
         - SVM with RBF kernel
         - Decision Tree
    4. For each model:
         - compute 5-fold CV accuracy on the training set
         - train on the full training set
         - evaluate on the held-out test set and save confusion matrix
         - track which model achieves the best test accuracy.
    5. Plot a bar chart comparing mean CV accuracy across all models.
    6. For the best model:
         - compute a learning curve using increasing training sizes
         - plot training vs cross-validation accuracy
         - estimate the minimum number of samples required to reach
           70% accuracy.
    """
    print("\n" + "=" * 70)
    print("DATASET 2: Model comparison and learning curve")
    print("=" * 70)

    pre = Preprocessor(DATASET_2_PATH, target_col="label")
    pre.load_data()
    pre.show_basic_info()
    preprocessor_all, feature_names = pre.build_numeric_pipeline()
    X_train, X_test, y_train, y_test = pre.train_test_split(test_size=0.25, stratify=True)

    evaluator = Evaluator(PLOTS_ROOT / "dataset2")

    # --- step 1: define candidate models -------------------------------------

    candidates = {
        "LogisticRegression": Classifier("logistic", preprocessor_all, feature_names),
        "KNN_5": Classifier("knn_5", preprocessor_all, feature_names),
        "RandomForest": Classifier("random_forest", preprocessor_all, feature_names),
        "SVM": Classifier("svm", preprocessor_all, feature_names),
        "DecisionTree": Classifier("decision_tree", preprocessor_all, feature_names),
    }

    best_name = None
    best_acc = -np.inf
    best_pipeline: Optional[Pipeline] = None

    # store mean CV accuracies for bar chart
    cv_means_dict: Dict[str, float] = {}

    # --- step 2: cross-validation + test evaluation --------------------------

    for name, clf in candidates.items():
        print(f"\n[Dataset 2] === {name} ===")
        # cross-validation on training set
        cv_scores = cross_val_score(
            clf.pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        print(
            f"  CV accuracy: mean={mean_cv:.4f}, "
            f"std={std_cv:.4f}"
        )

        # save mean CV accuracy for plotting later
        cv_means_dict[name] = mean_cv

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = evaluator.metrics(y_test, y_pred, name)
        evaluator.plot_confusion_matrix(
            y_test, y_pred, model_name=name, filename=f"confusion_{name}.png"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_pipeline = clf.pipeline

    print(
        f"\n[Dataset 2] Best test accuracy: {best_acc:.4f} "
        f"with model {best_name}"
    )

    # --- step 2.5: bar chart of CV accuracies --------------------------------

    evaluator.plot_cv_accuracy_bar(
        model_names=list(cv_means_dict.keys()),
        cv_means=list(cv_means_dict.values()),
        title="Dataset 2: Cross-validation accuracy by model",
        filename="cv_accuracy_bar.png",
    )

    # --- step 3: learning curve for best model -------------------------------

    if best_pipeline is None:
        return

    X, y = pre.get_features_and_target()

    print("\n[Dataset 2] Computing learning curve for best model...")

    # learning_curve requires that the number of samples used at each step
    # is at least `cv` (here 5) and at most n_samples * (cv-1)/cv.
    n_samples = X.shape[0]  # 400 in your dataset
    cv_folds = 5
    min_train_size = cv_folds  # smallest possible size = 5
    max_train_size = int(n_samples * (cv_folds - 1) / cv_folds)  # 400 * 4/5 = 320

    # Absolute train sizes: 5, 6, 7, ..., 320
    train_sizes = np.arange(min_train_size, max_train_size + 1, 1)

    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline,
        X,
        y,
        cv=cv_folds,
        train_sizes=train_sizes,
        scoring="accuracy",
        n_jobs=-1,
    )

    evaluator.plot_learning_curve(
        train_sizes,
        train_scores,
        test_scores,
        title=f"Learning curve ({best_name})",
        filename="learning_curve.png",
    )
    # Mean CV accuracy at each training size (average over folds).
    test_mean = np.mean(test_scores, axis=1)
    # Targets performance level we care about (70% accuracy).
    threshold = 0.70
    # Will stores the *first* training size that meets our stability condition.
    min_samples = None
    # How much drop below the threshold we still consider acceptable (1% here).
    tolerance = 0.01  # Allow 1% drop.
    # Loops over each training size n and its corresponding mean accuracy score.
    for i, (n, score) in enumerate(zip(train_sizes, test_mean)):
        # Prints a nice summary line for this point on the learning curve.
        print(f"  Train size {n:4d}: CV accuracy = {score:.4f}")
        
    # Only start looking for a candidate point if:
    #   - this point reaches the threshold.
    #   - we haven't already found a valid min_samples.
        if score >= threshold and min_samples is None:
            # Considers all accuracies from this point onwards.
            future_scores = test_mean[i:]

            # Checks if, after this point, the accuracy never drops.
            # more than 'tolerance' below the threshold.
            # i.e. if the *worst* future score is still ≥ (threshold - tolerance).
            if min(future_scores) >= (threshold - tolerance):
                # Then this n is our minimal stable sample size.
                min_samples = int(n)
    if min_samples is not None:

        print(
            f"\n[Dataset 2] Minimum samples to reach "
            f"{threshold * 100:.0f}% accuracy ≈ {min_samples}"
        )
    else:
        print(
            f"\n[Dataset 2] Accuracy did not reach {threshold * 100:.0f}% "
            f"in the explored training sizes."
        )


# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    run_dataset1_pipeline()
    run_dataset2_pipeline()
