import torch
import wandb
import numpy as np

from luminar.utils.training import (
    ConvolutionalLayerSpec,
    DEFAULT_CONV_LAYER_SHAPES,
    LuminarSequenceTrainingConfig,
    LuminarSequenceDataset
)
from luminar.sequence_classifier import LuminarSequence, LuminarSequenceAttention
from torch.utils.data import DataLoader, Subset, Dataset
from luminar.utils.evaluation import calculate_metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from collections import Counter
from torch.utils.data import WeightedRandomSampler

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
            code_artifact.add_file("data_hub/hub.py")
            code_artifact.add_file("data_hub/pipeline.py")
            code_artifact.add_file("data_hub/sequential_data_processor.py")
            code_artifact.add_file("training/train_luminar_sequence.py")
            wandb.log_artifact(code_artifact)

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n========== Fold {fold + 1}/{self.config.kfold} ==========")

            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)

            # If we have a class imbalance, we use weighted sampling and build a specific train
            # data loader for it.
            if getattr(self.config, "weighted_sampling", True):
                print("Applying weighted_sampling to this fold.")
                sample_weights, stats = self._compute_sampling_and_class_weights(train_subset)

                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_subset),  # epoch length = train fold size
                    replacement=True
                )
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    shuffle=False,  # must be False when sampler is set
                    drop_last=True,  # keep batch size stable
                    collate_fn=self.collate_fn
                )

                # Log the balance for visibility
                print(f"[Fold {fold + 1}] Weighted sampling ON | samples pos/neg = "
                      f"{stats['num_pos_samples']}/{stats['num_neg_samples']} | spans pos/neg = "
                      f"{stats['total_pos_spans']}/{stats['total_neg_spans']}")
                if self.log_to_wandb:
                    wandb.log({
                        f"fold_{fold}_num_pos_samples": stats["num_pos_samples"],
                        f"fold_{fold}_num_neg_samples": stats["num_neg_samples"],
                        f"fold_{fold}_total_pos_spans": stats["total_pos_spans"],
                        f"fold_{fold}_total_neg_spans": stats["total_neg_spans"],
                    })
            else:
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    collate_fn=self.collate_fn
                )
            eval_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate_fn)

            if self.use_experimental_attention:
                model = LuminarSequenceAttention(self.config).to(self.device)
            else:
                model = LuminarSequence(self.config).to(self.device)

            # (Optional) if the model/loss supports a pos_weight at span level, supply it:
            if getattr(self.config, "weighted_sampling", False):
                if hasattr(model, "set_pos_weight"):
                    _ = model.set_pos_weight(stats["pos_weight_spans"].to(self.device))
                elif hasattr(model, "pos_weight"):
                    model.pos_weight = stats["pos_weight_spans"].to(self.device)

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
                features = [f.to(self.device) for f in batch["features"]]
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
                features = [f.to(self.device) for f in batch["features"]]
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
                features = [f.to(self.device) for f in batch["features"]]
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

    def _compute_sampling_and_class_weights(self, subset: Subset):
        """
        Returns:
          sample_weights: torch.FloatTensor of len(subset) for WeightedRandomSampler
          stats: dict with counts and pos_weight tensors at both sample-level and span-level
        """
        sample_labels = []      # 1 if any span positive in the sample, else 0
        total_pos_spans = 0
        total_neg_spans = 0

        for i in range(len(subset)):
            item = subset[i]  # expects keys: "span_labels"
            span_labels = item["span_labels"]
            if not isinstance(span_labels, torch.Tensor):
                span_labels = torch.tensor(span_labels)

            pos_spans = (span_labels > 0.5).sum().item()
            neg_spans = (span_labels <= 0.5).sum().item()
            total_pos_spans += pos_spans
            total_neg_spans += neg_spans

            has_pos = 1 if pos_spans > 0 else 0
            sample_labels.append(has_pos)

        c = Counter(sample_labels)
        num_pos_samples = c.get(1, 0)
        num_neg_samples = c.get(0, 0)
        # avoid division by zero
        num_pos_samples = max(1, num_pos_samples)
        num_neg_samples = max(1, num_neg_samples)
        total_pos_spans = max(1, total_pos_spans)
        total_neg_spans = max(1, total_neg_spans)

        # inverse-frequency weights at sample level
        class_weight_samples = torch.tensor([1.0/num_neg_samples, 1.0/num_pos_samples], dtype=torch.float)
        sample_weights = torch.tensor([class_weight_samples[l] for l in sample_labels], dtype=torch.float)

        # pos_weight for BCEWithLogitsLoss at span level: N_neg / N_pos
        pos_weight_spans = torch.tensor([total_neg_spans / total_pos_spans], dtype=torch.float)
        pos_weight_samples = torch.tensor([num_neg_samples / num_pos_samples], dtype=torch.float)

        stats = {
            "num_pos_samples": int(c.get(1, 0)),
            "num_neg_samples": int(c.get(0, 0)),
            "total_pos_spans": int(total_pos_spans),
            "total_neg_spans": int(total_neg_spans),
            "pos_weight_spans": pos_weight_spans,
            "pos_weight_samples": pos_weight_samples,
        }
        return sample_weights, stats