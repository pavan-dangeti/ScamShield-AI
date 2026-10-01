import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
from scipy.sparse import hstack
from src.taxonomy import TACTICS

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    # Fill empty tactics with empty string
    df['tactics'] = df['tactics'].fillna('')
    # Convert tactics list to binary target columns
    for tactic in TACTICS:
        df[tactic] = df['tactics'].apply(lambda x: 1 if tactic in [t.strip() for t in x.split(',') if t.strip()] else 0)
    # Convert label to binary target (scam=1, legit=0)
    df['is_scam'] = df['label'].apply(lambda x: 1 if x == 'scam' else 0)
    return df
def train_models():
    print("Loading data...")
    df = load_data("data/scam_dataset.csv")
    
    # Split train and test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[['is_scam', 'language']]
    )
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    # Initialize vectorizers
    print("Fitting TF-IDF Vectorizers...")
    word_vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=5000, sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), max_features=10000, sublinear_tf=True
    )
    # Fit vectorizers on training text
    word_vectorizer.fit(train_df['text'])
    char_vectorizer.fit(train_df['text'])
    # Transform text
    X_train_word = word_vectorizer.transform(train_df['text'])
    X_train_char = char_vectorizer.transform(train_df['text'])
    X_train = hstack([X_train_word, X_train_char])
    X_test_word = word_vectorizer.transform(test_df['text'])
    X_test_char = char_vectorizer.transform(test_df['text'])
    X_test = hstack([X_test_word, X_test_char])
    # 1. Train Model A: Binary Scam Classifier
    print("Training Binary Scam Classifier (Model A)...")
    y_train_binary = train_df['is_scam']
    y_test_binary = test_df['is_scam']
    
    binary_model = LogisticRegression(class_weight='balanced', C=1.0, random_state=42, max_iter=1000)
    binary_model.fit(X_train, y_train_binary)
    # 2. Train Model B: Multi-Label Tactic Classifier
    print("Training Multi-Label Tactic Classifiers (Model B)...")
    tactic_models = {}
    for tactic in TACTICS:
        print(f"  Training classifier for tactic: {tactic}...")
        y_train_tactic = train_df[tactic]
        
        # Train a binary logistic regression model for this tactic
        model = LogisticRegression(class_weight='balanced', C=1.0, random_state=42, max_iter=1000)
        model.fit(X_train, y_train_tactic)
        tactic_models[tactic] = model
    # Save models and vectorizers
    os.makedirs("models", exist_ok=True)
    with open("models/vectorizers.pkl", "wb") as f:
        pickle.dump({"word": word_vectorizer, "char": char_vectorizer}, f)
    
    with open("models/binary_classifier.pkl", "wb") as f:
        pickle.dump(binary_model, f)
        
    with open("models/tactic_classifier.pkl", "wb") as f:
        pickle.dump(tactic_models, f)
        
    print("Models and vectorizers saved successfully.")
    # --- EVALUATION AND METRICS REPORT GENERATION ---
    print("Evaluating models and generating metrics report...")
    
    # Binary predictions
    binary_preds = binary_model.predict(X_test)
    binary_probs = binary_model.predict_proba(X_test)[:, 1]
    
    # Tactic predictions
    tactic_preds = {}
    for tactic, model in tactic_models.items():
        tactic_preds[tactic] = model.predict(X_test)
    # Build report strings
    report = []
    report.append("# ScamShield Model Metrics & Evaluation Report\n")
    report.append("This report breaks down the classification metrics for the ScamShield binary scam classifier and the multi-label tactic classifiers across different languages and scripts.\n")
    # Global Binary Metrics
    p, r, f, s = precision_recall_fscore_support(y_test_binary, binary_preds, average='binary')
    report.append("## Global Scam Classifier Performance")
    report.append("| Metric | Score |")
    report.append("|---|---|")
    report.append(f"| Precision | {p:.4f} |")
    report.append(f"| Recall | {r:.4f} |")
    report.append(f"| F1-Score | {f:.4f} |")
    report.append(f"| Support | {s} |")
    report.append("")
    # Global Tactic Metrics
    report.append("## Global Tactic Classifier Performance")
    report.append("| Tactic | Precision | Recall | F1-Score | Support |")
    report.append("|---|---|---|---|---|")
    for tactic in TACTICS:
        tp, tr, tf, ts = precision_recall_fscore_support(test_df[tactic], tactic_preds[tactic], average='binary', zero_division=0)
        report.append(f"| {tactic.replace('_', ' ').title()} | {tp:.4f} | {tr:.4f} | {tf:.4f} | {test_df[tactic].sum()} |")
    report.append("")
    # Per Language/Script Breakdown
    report.append("## Performance Breakdown by Language and Script")
    report.append("Shows where the model performs well and identifies languages or scripts that require more training data.\n")
    
    # Group by language and script
    groups = test_df.groupby(['language', 'script'])
    
    for (lang, script), group_df in groups:
        report.append(f"### {lang.title()} ({script.title()})")
        report.append(f"Total test samples: {len(group_df)}")
        
        # Filter test indices
        group_indices = group_df.index
        # Get relative position in test_df
        test_positions = [test_df.index.get_loc(idx) for idx in group_indices]
        
        y_true_grp = group_df['is_scam']
        y_pred_grp = binary_preds[test_positions]
        
        # Check if we have positive/negative samples in this group to avoid NaN warning
        scam_count = y_true_grp.sum()
        legit_count = len(y_true_grp) - scam_count
        
        report.append(f"- **Scam Samples**: {scam_count}")
        report.append(f"- **Legit Samples**: {legit_count}")
        
        if scam_count > 0 and legit_count > 0:
            gp, gr, gf, gs = precision_recall_fscore_support(y_true_grp, y_pred_grp, average='binary', zero_division=0)
            report.append("")
            report.append("| Classifier | Precision | Recall | F1-Score |")
            report.append("|---|---|---|---|")
            report.append(f"| **Scam Classifier** | {gp:.4f} | {gr:.4f} | {gf:.4f} |")
            
            # Tactic metrics for this group
            for tactic in TACTICS:
                grp_y_true = group_df[tactic]
                grp_y_pred = tactic_preds[tactic][test_positions]
                
                # Check support
                tactic_support = grp_y_true.sum()
                if tactic_support > 0:
                    tp, tr, tf, ts = precision_recall_fscore_support(grp_y_true, grp_y_pred, average='binary', zero_division=0)
                    report.append(f"| Tactic: {tactic.replace('_', ' ').title()} | {tp:.4f} | {tr:.4f} | {tf:.4f} | (Support: {tactic_support})")
                else:
                    report.append(f"| Tactic: {tactic.replace('_', ' ').title()} | N/A | N/A | N/A | (Support: 0)")
            report.append("\n")
        else:
            report.append("*N/A: Group does not contain both scam and legit samples for robust evaluation.*\n")
    # Write report file
    report_content = "\n".join(report)
    with open("models/metrics_report.md", "w", encoding="utf-8") as report_file:
        report_file.write(report_content)
    
    print("Report generated and saved to models/metrics_report.md")
if __name__ == "__main__":
    train_models()
