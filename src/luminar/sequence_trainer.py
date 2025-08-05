import torch
import wandb
import numpy as np

from luminar.utils.training import (
    ConvolutionalLayerSpec, 
    DEFAULT_CONV_LAYER_SHAPES, 
    LuminarSequenceDataset,
    LuminarSequenceTrainingConfig
)
from luminar.sequence_classifier import LuminarSequence, LuminarSequenceAttention
from torch.utils.data import DataLoader, Subset, Dataset
from luminar.utils.evaluation import calculate_metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

class LuminarSequenceTrainer:

    def __init__(self,
                 train_dataset : LuminarSequenceDataset,
                 test_data_loader : DataLoader,
                 collate_fn : callable,
                 config : LuminarSequenceTrainingConfig,
                 log_to_wandb : bool = False, 
                 device : str = "",
                 use_experimental_attention : bool = False):
        self.config = config
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        self.test_data_loader = test_data_loader
        self.log_to_wandb = log_to_wandb
        self.device = device
        self.use_experimental_attention = use_experimental_attention

    def train(self):
        """
        Starts the training of a LuminarSequence with KFold CV and test evaluation.
        Reports to wandb and stores the models
        """
        kf = KFold(n_splits=self.config.kfold, shuffle=True, random_state=42)
        fold_metrics = []
        indices = list(range(len(self.train_dataset)))
        best_model = None

        # Initialize wandb fold metrics table
        if self.log_to_wandb:
            fold_table_columns = ["fold"] + list(self._evaluate_test(LuminarSequence(self.config).to(self.device)).keys())
            fold_table = wandb.Table(columns=fold_table_columns)

            # Also store relevant code files
            code_artifact = wandb.Artifact(name="luminar_sequence_code", type="code")
            code_artifact.add_dir("luminar")
            code_artifact.add_file("training/train_luminar_sequence.py")
            wandb.log_artifact(code_artifact)

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n========== Fold {fold + 1}/{self.config.kfold} ==========")

            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
            eval_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn)

            if self.use_experimental_attention:
                model = LuminarSequenceAttention(self.config).to(self.device)
            else:
                model = LuminarSequence(self.config).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

            model = self._train_and_evaluate(
                model, train_loader, eval_loader, optimizer, self.config.max_epochs, fold, self.config.early_stopping_patience
            )

            # Evaluate and collect fold metrics
            metrics = self._evaluate_test(model)
            # Update best model if it's the first or if it has a higher F1 score than all previous ones
            if best_model is None or metrics["f1_score"] > max(m["f1_score"] for m in fold_metrics):
                best_model = model
            fold_metrics.append(metrics)

            print(f"\nFold {fold + 1} Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Add to wandb table
            if self.log_to_wandb:
                fold_table.add_data(fold, *metrics.values())

        # Log the full table to wandb
        if self.log_to_wandb:
            wandb.log({"kfold_test_metrics": fold_table})

        # Average metrics across folds
        aggregated_metrics = {}
        for m in fold_metrics:
            for k, v in m.items():
                aggregated_metrics.setdefault(k, []).append(v)

        avg_metrics = {}
        print(f"\n========== K-Fold Average Metrics ==========")
        for k, values in aggregated_metrics.items():
            avg_val = sum(values) / len(values)
            avg_metrics[k] = avg_val
            print(f"  {k}: {avg_val:.4f}")
            if self.log_to_wandb:
                wandb.summary[f"test_avg_{k}"] = avg_val

        return (avg_metrics, best_model)

    def _train_and_evaluate(self, 
                           model : LuminarSequence, 
                           train_loader : DataLoader, 
                           eval_loader : DataLoader, 
                           optimizer, 
                           epochs : int,
                           fold : int, 
                           patience : int = 3):
        """
        Trains the given model an the given train and eval loader and returns the best performing model state.
        """

        best_eval_loss = float("inf")
        best_train_loss = None
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                features = batch["features"].to(self.device)
                sentence_spans = batch["sentence_spans"]
                span_labels = batch["span_labels"]

                optimizer.zero_grad()
                output = model(features, sentence_spans, span_labels=span_labels)
                loss = output.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            # Evaluation
            avg_eval_loss = self._evaluate(model, eval_loader)
            print(f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
            if self.log_to_wandb:
                wandb.log({
                    f"fold_{fold}_train_loss": avg_train_loss,
                    f"fold_{fold}_eval_loss": avg_eval_loss,
                })

            # Early Stopping & Checkpoint
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_train_loss = avg_train_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

        print(f"\nBest Eval Loss: {best_eval_loss:.4f} | Best Train Loss: {best_train_loss:.4f}")
        return model

    def _evaluate(self, model : LuminarSequence, eval_loader : DataLoader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                features = batch["features"].to(self.device)
                sentence_spans = batch["sentence_spans"]
                span_labels = batch["span_labels"]

                output = model(features, sentence_spans, span_labels=span_labels)
                loss = output.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(eval_loader)
        return avg_loss

    def _evaluate_test(self, model: LuminarSequence):
        model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_data_loader:
                features = batch["features"].to(self.device)
                sentence_spans = batch["sentence_spans"]
                span_labels = batch["span_labels"]

                output = model(features, sentence_spans)
                probs = torch.sigmoid(output.logits).view(-1).cpu().numpy()
                labels = torch.cat([torch.tensor(lbl, dtype=torch.int) for lbl in span_labels]).cpu().numpy()

                all_scores.extend(probs)
                all_labels.extend(labels)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        metrics = calculate_metrics(all_labels, all_scores, threshold=0.5)
        return metrics