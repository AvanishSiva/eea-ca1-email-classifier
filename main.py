# This is the main file: The controller. All methods will directly or indirectly be called here.
#
# to support multi-label classification via two design decisions:
#   Design Decision 1: Chained Multi-Output Classification
#   Design Decision 2: Hierarchical Modelling

from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from modelling.chained_multi_output import run_chained
from modelling.hierarchical_modelling import run_hierarchical
from model.randomforest import RandomForest
from model.SGD import SGD
import random

seed = 0
random.seed(seed)
np.random.seed(seed)



def load_data():
    """Load the input data."""
    df = get_input_data()
    return df


def preprocess_data(df):
    """De-duplicate, remove noise, translate."""
    df = de_duplication(df)
    df = noise_remover(df)
    # Translation is skipped in this environment
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embd(df: pd.DataFrame):
    """Get TF-IDF embeddings."""
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create the Data encapsulation object."""
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    """Original modelling — single y2 classification via uniform interface."""
    model_predict(data, df, name)



def run_design_decision_1(X, group_df, group_name):
    """
    DD1: ONE model instance trained on the fully combined label (y2+y3+y4).
    Evaluated at three chain levels. Accuracy cascades — a wrong y2
    prediction makes deeper levels wrong too.
    """
    print("\n" + "=" * 70)
    print(f"  DESIGN DECISION 1: Chained Multi-Output — Group: {group_name}")
    print("=" * 70)

    all_results = {}
    for model_class, m_name in [(RandomForest, "RandomForest"), (SGD, "SGD")]:
        result = run_chained(X.copy(), group_df.copy(), model_class, m_name)
        if result:
            all_results[m_name] = result

    return all_results



def run_design_decision_2(X, group_df, group_name):
    """
    DD2: Tree of model instances. Level 1 classifies y2. For each y2
    class, a new model classifies y3 on filtered data. For each (y2,y3)
    pair, another new model classifies y4. Number of models grows with
    number of classes.
    """
    print("\n" + "=" * 70)
    print(f"  DESIGN DECISION 2: Hierarchical Modelling — Group: {group_name}")
    print("=" * 70)

    all_results = {}
    for model_class, m_name in [(RandomForest, "RandomForest"), (SGD, "SGD")]:
        result = run_hierarchical(X.copy(), group_df.copy(), model_class, m_name)
        if result:
            all_results[m_name] = result

    return all_results



def print_comparison(dd1_results, dd2_results, group_name):
    """Print a side-by-side comparison of both design decisions."""
    print("\n" + "=" * 70)
    print(f"  COMPARISON SUMMARY — Group: {group_name}")
    print("=" * 70)

    print("\n  Design Decision 1: Chained Multi-Output")
    print(f"  {'Model':<20} {'Level 1 (y2)':>14} {'Level 2 (y2+y3)':>16} {'Level 3 (y2+y3+y4)':>20}")
    print("  " + "-" * 72)
    for m_name, res in dd1_results.items():
        l1 = res.get("Level 1", 0)
        l2 = res.get("Level 2", 0)
        l3 = res.get("Level 3", 0)
        print(f"  {m_name:<20} {l1:>14.4f} {l2:>16.4f} {l3:>20.4f}")

    print("\n  Design Decision 2: Hierarchical Modelling")
    print(f"  {'Model':<20} {'Level 1 (y2)':>14} {'Level 2 avg (y3)':>16} {'Level 3 avg (y4)':>20}")
    print("  " + "-" * 72)
    for m_name, res in dd2_results.items():
        l1 = res.get("l1_acc", 0) or 0
        l2_accs = [d["accuracy"] for d in res.get("l2_details", [])]
        l3_accs = [d["accuracy"] for d in res.get("l3_details", [])]
        l2_avg = sum(l2_accs) / len(l2_accs) if l2_accs else 0
        l3_avg = sum(l3_accs) / len(l3_accs) if l3_accs else 0
        print(f"  {m_name:<20} {l1:>14.4f} {l2_avg:>16.4f} {l3_avg:>20.4f}")

    print("\n  Note: DD1 accuracy cascades — a wrong y2 makes deeper levels wrong.")
    print("  DD2 averages are across per-class sub-models at each level.")
    print("=" * 70)



if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print("\n" + "#" * 70)
        print(f"# GROUP: {name}")
        print("#" * 70)

        X, group_df = get_embd(group_df)

        print("\n--- Original: Single-label classification (y2) ---")
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)

        dd1 = run_design_decision_1(X, group_df, name)

        dd2 = run_design_decision_2(X, group_df, name)

        if dd1 and dd2:
            print_comparison(dd1, dd2, name)
