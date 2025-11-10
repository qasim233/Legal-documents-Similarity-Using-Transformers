# Legal Clause Similarity Detection using Deep Learning

## Overview

This project implements two baseline NLP architectures for detecting semantic similarity between legal clauses without using pre-trained transformers or fine-tuned legal models. The solution is built entirely using **PyTorch**.

## Problem Statement

Legal documents contain clauses written in formal, structured language that often express the same legal principles in different ways. This project addresses the challenge of identifying semantic equivalence and contextual relatedness between legal clauses through two key dimensions:

1. **Semantic Equivalence**: Whether two clauses express the same legal principle
2. **Contextual Relatedness**: Whether two clauses address related legal concepts

## Dataset Structure

The dataset consists of multiple CSV files where:
- Each CSV file name represents a distinct clause category (e.g., acceleration, access-to-information)
- Each file contains clause texts and their corresponding clause type labels
- Categories are used to generate positive pairs (similar) and negative pairs (dissimilar)

**Expected Dataset Structure:**
```
legal_clauses_dataset/
    ├── acceleration.csv
    ├── access-to-information.csv
    ├── accounting-terms.csv
    └── ... (other category files)
```

## Model Architectures

### Model 1: Siamese Network with BiLSTM and Attention

**Architecture Components:**
- **Embedding Layer**: Converts words to dense vector representations
- **Bidirectional LSTM**: Captures sequential context from both directions (using `nn.LSTM`)
- **Custom Attention Mechanism**: Focuses on legally important terms (custom `nn.Module`)
- **Siamese Structure**: Shared weights ensure consistent encoding
- **Classification Head**: Multiple similarity features for binary classification

**Key Features:**
- Parameter efficient due to weight sharing
- Effective for sequential pattern recognition
- Custom attention highlights important legal terms
- Implemented as PyTorch `nn.Module`

### Model 2: Dual Encoder with Multi-Head Attention

**Architecture Components:**
- **Embedding Layer with Positional Encoding**: Word and position information (using `nn.Embedding`)
- **Multi-Head Self-Attention**: Captures multiple semantic aspects simultaneously (using `nn.MultiheadAttention`)
- **Feed-Forward Networks**: Non-linear transformations
- **Dual Encoder**: Separate encoders for specialized learning
- **Classification Head**: Cosine similarity and interaction features

**Key Features:**
- More expressive with separate encoders
- Multi-head attention captures complex relationships
- Inspired by Transformer architecture

## Evaluation Metrics

The models are evaluated using comprehensive NLP metrics:

1. **Accuracy**: Measures overall classification correctness
2. **Precision**: Out of predicted similar pairs, how many are truly similar
3. **Recall**: Out of truly similar pairs, how many were identified
4. **F1-Score**: Harmonic mean of Precision and Recall
5. **ROC-AUC**: Model's ability to separate similar vs dissimilar pairs
6. **PR-AUC**: Precision-Recall AUC for imbalanced dataset handling

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

## Installation

### Option 1: CPU Version
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

### Option 2: GPU Version (Recommended)
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Option 3: Using requirements.txt
```bash
pip install -r requirements.txt
```

### Verify Installation
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## Usage

### 1. Prepare Your Dataset

Place your legal clause CSV files in a directory (default: `./legal_clauses_dataset/`). Each CSV should have columns for clause text and clause type.

### 2. Configure Parameters

Update the configuration in the notebook:
```python
DATA_DIRECTORY = './legal_clauses_dataset'  # dataset path
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EPOCHS = 30
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Auto-detect GPU
```

### 3. Run the Notebook

Execute all cells in `model_development.ipynb` sequentially. The notebook will:
- Load and preprocess the data
- Build both PyTorch model architectures
- Train with early stopping and learning rate scheduling
- Evaluate on test set
- Generate comprehensive visualizations
- Compare model performance

### 4. Model Training

Both models are trained with:
- Early stopping (patience=5)
- Learning rate reduction on plateau
- Model checkpointing (saves best weights)
- Validation monitoring

### 5. Evaluation and Visualization

The notebook generates:
- Training history plots (loss, accuracy, AUC)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Side-by-side model comparison

## Project Structure

```
Assignment_2/
├── model_development.ipynb    # Main notebook with complete implementation
├── README.md                   # This file
└── model/                      # Directory for saved models
    ├── Siamese_BiLSTM_Attention_best.h5
    └── Dual_Encoder_MultiHead_Attention_best.h5
```

## Code Organization

The implementation follows a modular, object-oriented design:

1. **LegalClauseDataLoader**: Handles data loading and pair generation
2. **TextPreprocessor**: Manages tokenization and sequence processing
3. **AttentionLayer**: Custom attention mechanism
4. **SiameseAttentionModel**: Siamese BiLSTM architecture
5. **MultiHeadAttentionEncoder**: Dual encoder architecture
6. **ModelTrainer**: Training, evaluation, and visualization utilities
7. **compare_models**: Model comparison functionality

## Key Features

✓ **Comprehensive Comments**: Every function and class is thoroughly documented
✓ **Human-Readable Code**: Clear variable names and logical structure
✓ **Modular Design**: Easy to extend and modify
✓ **Complete Pipeline**: From data loading to model comparison
✓ **Visualization**: Multiple plots for performance analysis
✓ **TensorFlow Native**: Built entirely with TensorFlow/Keras
✓ **No Pre-trained Models**: All architectures built from scratch

## Model Training Tips

1. **For Small Datasets**: Reduce model complexity (lower LSTM_UNITS, EMBEDDING_DIM)
2. **For Large Datasets**: Increase BATCH_SIZE for faster training
3. **For Better Accuracy**: Increase VOCAB_SIZE and MAX_SEQUENCE_LENGTH
4. **For Faster Training**: Use GPU acceleration (automatically detected)

## Expected Output

After running the complete pipeline, you will get:

1. **Training Metrics**: Loss, accuracy, precision, recall, AUC per epoch
2. **Test Evaluation**: All metrics on held-out test set
3. **Visualizations**: 
   - Training curves for both models
   - Confusion matrices
   - ROC and PR curves
   - Comparative bar charts
4. **Model Comparison Table**: Side-by-side metric comparison
5. **Saved Models**: Best model weights for inference

## Inference

Use the `predict_similarity()` function to test new clause pairs:

```python
clause_1 = "Your first legal clause here"
clause_2 = "Your second legal clause here"

similarity = predict_similarity(
    clause_1, clause_2, 
    siamese_model, 
    preprocessor, 
    "Siamese BiLSTM"
)
```

## Performance Considerations

- **Training Time**: Depends on dataset size and hardware (GPU recommended)
- **Memory Usage**: Scales with VOCAB_SIZE and BATCH_SIZE
- **Inference Speed**: Both models provide fast predictions (<100ms per pair)

## Limitations

1. No pre-trained embeddings used (trainable embeddings only)
2. Limited to binary classification (similar/dissimilar)
3. Requires sufficient training data for good performance
4. Legal domain-specific terminology needs adequate representation

## Future Enhancements

1. Incorporate legal domain word embeddings
2. Add multi-class similarity levels (e.g., highly similar, moderately similar)
3. Implement cross-validation for robust evaluation
4. Add explainability features (attention visualization)
5. Ensemble both models for improved accuracy

## References

- **Siamese Networks**: Learning semantic similarity through shared weights
- **Attention Mechanisms**: Focusing on important words in sequences
- **BiLSTM**: Bidirectional context capture for NLP
- **Multi-Head Attention**: Parallel attention for multiple semantic aspects
- **Legal NLP**: Semantic similarity in legal document analysis

## Author

Deep Learning Assignment 2 - Legal Clause Similarity Detection

## License

This project is for educational purposes.
