import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import logging
from datasets import load_dataset

logging.set_verbosity_error()
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


EPOCHS = 10

SEED_VAL = 2022
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# Define hyperparameters grid
HYPERPARAMETERS_GRID = [
    {
        "optimizer": "AdamW",
        "lr": 5e-5,
        "eps": 1e-5,
    },
    {
        "optimizer": "RMSprop",
        "lr": 0.01,
        "alpha": 0.99,
        "eps": 1e-5,
    },
    {
        "optimizer": "NAG",
        "lr": 5e-5,
    },
    {
        "optimizer": "SGD",
        "lr": 0.01,
    },
    {
        "optimizer": "SGD Momentum",
        "lr": 5e-5,
        "momentum": 0.001,
    },
]



def load_fin_dataset():
    """Load the financial phrasebank dataset."""
    # load dataset from huggingface datasets library
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    df = pd.DataFrame(dataset['train'])
    rename_cols = {'sentence': 'NewsHeadline', 'label': 'sentiment'}
    df = df.rename(columns=rename_cols)
    # 'positive' (2), 'negative' (0) or 'neutral' (1)
    df['category'] = df['sentiment'].apply(lambda x: "negative" if x == 0 else ("neutral" if x == 1 else "positive"))
    return df


def encode_sentiments_values(df):
    """Label encode the sentiment values."""
    possible_sentiments = df.sentiment.unique()
    sentiment_dict = {}
    # Enumerate method adds counter to an iterable and returns it.
    for index, possible_sentiment in enumerate(possible_sentiments):
        sentiment_dict[possible_sentiment] = index
    # Encode all the sentiment values
    df["label"] = df.sentiment.replace(sentiment_dict)
    return df, sentiment_dict


def tokenize_encode_dataset(dataset):
    """Tokenize and encode the input data using the `batch_encode_plus` method of the tokenizer."""
    # Get the BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    encoded_data = tokenizer.batch_encode_plus(
        dataset.NewsHeadline.values,
        # Can be set to ‘tf’ or ‘pt’ to return respectively TensorFlow tf.constant or PyTorch
        # torch.Tensor for the input data instead of a list of python integers.
        return_tensors="pt",
        # Adds special tokens (e.g., [CLS], [SEP]) required by the transformer model
        add_special_tokens=True,
        # Return attention masks along with the encoded data to indicate which tokens should be attended to by the model
        return_attention_mask=True,
        # Pad sequences to the maximum length specified by `max_length`
        pad_to_max_length=True,
        # Maximum length of the sequences after padding, longer sequences will be truncated to this length
        max_length=150,
    )
    return encoded_data


def create_data_loader(encoded_data, y):
    """Create a DataLoader for the training and validation data."""
    input_ids, attention_masks = encoded_data["input_ids"], encoded_data["attention_mask"]
    sentiment_labels = torch.tensor(y)
    dataset = TensorDataset(input_ids, attention_masks, sentiment_labels)

    batch_size = 32
    # Torch DataLoader
    dataloader = DataLoader(
        # Samples elements randomly.
        dataset, sampler=RandomSampler(dataset), batch_size=batch_size
    )

    return dataloader


def evaluate(validation_data):
    """Evaluate the model on the validation data."""
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in validation_data:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss, logits = outputs[0], outputs[1]
        loss_val_total += loss.item()

        # Convert logits to predicted class labels
        preds = torch.argmax(logits, axis=1).cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.extend(preds)
        true_vals.extend(label_ids)

    loss_val_avg = loss_val_total / len(validation_data)
    accuracy = accuracy_score(true_vals, predictions)
    return loss_val_avg, accuracy, predictions, true_vals


def model_training(train, val) -> list:
    """Train the model using the hyperparameters grid."""
    results = []
    for params in tqdm(HYPERPARAMETERS_GRID):
        optimizer_name = params["optimizer"]

        print(f"Optimizer Name: {optimizer_name}")
        del params["optimizer"]

        if optimizer_name == "AdamW":
            optimizer = AdamW(model.parameters(), **params)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), **params)
        elif optimizer_name == "NAG":
            optimizer = torch.optim.SGD(model.parameters(), **params)
            optimizer.defaults["nesterov"] = True
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), **params)
        elif optimizer_name == "SGD Momentum":
            optimizer = torch.optim.SGD(model.parameters(), **params)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * EPOCHS
        )

        for epoch in tqdm(range(1, EPOCHS + 1)):
            model.train()
            loss_train_total = 0
            progress_bar = tqdm(
                train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
            )

            for batch in progress_bar:
                model.zero_grad()
                batch = tuple(b.to(device) for b in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                }

                outputs = model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                # Gradient Clipping is done to restrict the values of the gradient, to prevent the model from exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix(
                    {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
                )

            # Save model after each epoch
            torch.save(model.state_dict(), f"finetuned_BERT_{optimizer_name}_epoch_{epoch}.model")
            tqdm.write(f"\nEpoch {epoch}")
            # Print training statistics
            loss_train_avg = loss_train_total / len(train)
            tqdm.write(f"Training loss: {loss_train_avg}")

            # Evaluate model on validation set
            val_loss, accuracy, predictions, true_vals = evaluate(val)
            val_f1 = f1_score(predictions, true_vals, average="weighted")
            tqdm.write(f"Validation loss: {val_loss}")
            tqdm.write(f"F1 Score (Weighted): {val_f1}")
            tqdm.write(f"Accuracy: {accuracy}")

            results.append({
                "optimizer": optimizer_name,
                "learning_rate": params["lr"],
                "momentum": params.get("momentum", None),
                "alpha": params.get("alpha", None),
                "beta1": params.get("beta1", None),
                "beta2": params.get("beta2", None),
                "epsilon": params.get("epsilon", None),
                "epoch": epoch,
                "training_loss": loss_train_avg,
                "validation_loss": val_loss,
                "accuracy": accuracy
            })

    return results


def plot_results(lst):
    # Create DataFrame from results
    results_df = pd.DataFrame(lst)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="epoch", y="training_loss", hue="optimizer")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Optimizer")
    plt.grid(True)
    plt.show()

    # Plot validation loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="epoch", y="validation_loss", hue="optimizer")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Optimizer")
    plt.grid(True)
    plt.show()


# load dataset
financial_data = load_fin_dataset()
# Encode the sentiment column
financial_data, sentiment_dict = encode_sentiments_values(financial_data)

# Create training and validation data, training set as 80% and test set as 20%
X_train, X_val, y_train, y_val = train_test_split(
    financial_data.index.values,
    financial_data.label.values,
    test_size=0.20,
    random_state=2022,
    stratify=financial_data.label.values,
)

X_train, X_val = financial_data.loc[X_train], financial_data.loc[X_val]

# Encode the Training and Validation Data
encoded_data_train, encoded_data_val = tokenize_encode_dataset(X_train), tokenize_encode_dataset(X_val)
dataloader_train = create_data_loader(encoded_data_train, y_train)
dataloader_validation = create_data_loader(encoded_data_val, y_val)

# Load the pre-trained FinBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", num_labels=len(sentiment_dict)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# model training with hyperparameters grid
model_results = model_training(dataloader_train, dataloader_validation)
# plot results after training
plot_results(model_results)
