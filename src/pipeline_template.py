"""
Complete pipeline for training and creating submissions - BountyBench Style

This script handles:
1. Loading training and validation data
2. Extracting features
3. Training a model
4. Evaluating on validation set
5. Making predictions on test set
6. Creating submission file

Usage:
    # Train and evaluate
    python pipeline.py train --task A
    
    # Create submission for Kaggle
    python pipeline.py submit --task A --test-data data/test_A.parquet
    
    # Train with specific model
    python pipeline.py train --task A --model-type gradient_boosting
    
    # Debug mode with verbose logging
    python pipeline.py train --task A --debug
"""

from pathlib import Path
from typing import Optional
import pickle

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger
import pandas as pd
import numpy as np

from data_loader import TaskDataLoader
from features import extract_features_from_dataframe
from models import get_model
from evaluate import evaluate, print_results
from create_submission import create_submission_from_arrays

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

# Initialize Rich console for beautiful output
console = Console()
app = typer.Typer(
    name="SemEval Task 13 Pipeline",
    help="Production-ready pipeline for ML-generated code detection",
    add_completion=False,
)


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag"""
    logger.remove()  # Remove default handler
    
    if debug:
        # Verbose logging for debugging
        logger.add(
            "logs/pipeline_debug.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
        )
        logger.add(
            lambda msg: console.print(msg, style="dim"),
            format="{message}",
            level="DEBUG",
        )
    else:
        # Clean logging for production
        logger.add(
            "logs/pipeline.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="10 MB",
        )
        logger.add(
            lambda msg: console.print(msg),
            format="{message}",
            level="INFO",
            filter=lambda record: record["level"].name == "INFO",
        )


@app.command()
def info(
    task: str = typer.Option(..., "--task", "-t", help="Task identifier: A, B, or C"),
) -> None:
    """
    Show dataset statistics and model information
    
    Examples:
        python pipeline.py info --task A
    """
    console.rule(f"[bold blue]Task {task} Information")
    
    try:
        # Load data
        loader = TaskDataLoader(task)
        train_df = loader.load_split('train')
        val_df = loader.load_split('validation')
        
        # Dataset stats
        table = Table(title=f"Task {task} Dataset Statistics")
        table.add_column("Split", style="cyan")
        table.add_column("Samples", style="magenta")
        table.add_column("Classes", style="green")
        
        table.add_row("Training", f"{len(train_df):,}", str(train_df['label'].nunique()))
        table.add_row("Validation", f"{len(val_df):,}", str(val_df['label'].nunique()))
        
        console.print(table)
        
        # Class distribution
        console.print("\n[bold]Class Distribution (Training):[/bold]")
        for label, count in train_df['label'].value_counts().items():
            pct = count / len(train_df) * 100
            console.print(f"  Class {label}: {count:,} ({pct:.1f}%)")
        
        # Check for saved models
        model_dir = Path("models")
        if model_dir.exists():
            models = list(model_dir.glob(f"task_{task}_*.pkl"))
            if models:
                console.print(f"\n[bold green]✓ Found {len(models)} saved model(s)[/bold green]")
                for model_path in models:
                    console.print(f"  • {model_path.name}")
            else:
                console.print("\n[yellow]No saved models found. Run 'train' first.[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    task: str = typer.Option(..., "--task", "-t", help="Task identifier: A, B, or C"),
    model_type: str = typer.Option(
        "random_forest",
        "--model-type",
        "-m",
        help="Model type: logistic_regression, random_forest, gradient_boosting",
    ),
    save_model: bool = typer.Option(True, "--save/--no-save", help="Save trained model"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """
    Train a baseline model on the specified task
    
    Examples:
        python pipeline.py train --task A
        python pipeline.py train --task B --model-type gradient_boosting
        python pipeline.py train --task A --debug
    """
    setup_logging(debug)
    
    console.rule(f"[bold blue]Training Pipeline - Task {task}")
    logger.info(f"Starting training pipeline for Task {task}")
    
    try:
        # Step 1: Load data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading data...", total=None)
            
            logger.debug(f"Initializing TaskDataLoader for task {task}")
            loader = TaskDataLoader(task)
            train_df = loader.load_split('train')
            val_df = loader.load_split('validation')
            
            logger.debug(f"Train set: {len(train_df)} samples")
            logger.debug(f"Validation set: {len(val_df)} samples")
        
        console.print(f"✓ Loaded {len(train_df):,} training samples", style="green")
        console.print(f"✓ Loaded {len(val_df):,} validation samples", style="green")
        
        # Step 2: Extract features
        console.print("\n[bold]Extracting features...[/bold]")
        logger.info("Starting feature extraction")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Training set features...", total=None)
            X_train = extract_features_from_dataframe(train_df)
            
            progress.add_task("Validation set features...", total=None)
            X_val = extract_features_from_dataframe(val_df)
        
        y_train = train_df['label']
        y_val = val_df['label']
        
        logger.debug(f"Extracted {X_train.shape[1]} features")
        console.print(f"✓ Extracted {X_train.shape[1]} features", style="green")
        
        # Step 3: Train model
        console.print(f"\n[bold]Training {model_type} model...[/bold]")
        logger.info(f"Initializing {model_type} model")
        
        model = get_model(model_type)
        model.fit(X_train, y_train)
        
        # Step 4: Evaluate
        console.print("\n[bold]Evaluating on validation set...[/bold]")
        logger.info("Running validation evaluation")
        
        y_pred = model.predict(X_val)
        val_metrics = evaluate(y_val, y_pred, detailed=True)
        
        # Display results table
        results_table = Table(title="Validation Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Score", style="magenta")
        
        results_table.add_row("Macro F1", f"{val_metrics['macro_f1']:.4f}")
        results_table.add_row("Micro F1", f"{val_metrics['micro_f1']:.4f}")
        results_table.add_row("Weighted F1", f"{val_metrics['weighted_f1']:.4f}")
        
        console.print(results_table)
        
        # Step 5: Save model
        if save_model:
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f'task_{task}_{model_type}.pkl'
            
            logger.info(f"Saving model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            console.print(f"\n✓ Model saved to {model_path}", style="green bold")
        
        console.rule("[bold green]Training Complete!")
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def submit(
    task: str = typer.Option(..., "--task", "-t", help="Task identifier: A, B, or C"),
    test_data: Path = typer.Option(..., "--test-data", help="Path to test data file"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Path to saved model"),
    model_type: str = typer.Option(
        "random_forest",
        "--model-type",
        "-m",
        help="Model type (if training from scratch)",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output submission path"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """
    Create submission file for Kaggle
    
    Examples:
        python pipeline.py submit --task A --test-data data/test_A.parquet
        python pipeline.py submit --task A --test-data data/test_A.parquet --model-path models/my_model.pkl
    """
    setup_logging(debug)
    
    console.rule(f"[bold blue]Submission Pipeline - Task {task}")
    logger.info(f"Starting submission pipeline for Task {task}")
    
    try:
        # Step 1: Load or train model
        if model_path and model_path.exists():
            console.print(f"Loading model from {model_path}...")
            logger.info(f"Loading model: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            console.print("✓ Model loaded", style="green")
        else:
            console.print("[yellow]No model found, training from scratch...[/yellow]")
            logger.warning("Model not found, initiating training")
            
            # Train model inline
            train(task=task, model_type=model_type, save_model=True, debug=debug)
            
            # Load the newly trained model
            model_path = Path('models') / f'task_{task}_{model_type}.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # Step 2: Load test data
        console.print(f"\nLoading test data from {test_data}...")
        logger.info(f"Loading test data: {test_data}")
        
        if not test_data.exists():
            console.print(f"[bold red]Error:[/bold red] Test data not found: {test_data}")
            raise typer.Exit(code=1)
        
        if test_data.suffix == '.parquet':
            test_df = pd.read_parquet(test_data)
        else:
            test_df = pd.read_csv(test_data)
        
        console.print(f"✓ Loaded {len(test_df):,} test samples", style="green")
        logger.debug(f"Test set shape: {test_df.shape}")
        
        # Step 3: Extract features and predict
        console.print("\n[bold]Extracting features and making predictions...[/bold]")
        logger.info("Starting feature extraction for test set")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing test set...", total=None)
            X_test = extract_features_from_dataframe(test_df, 'code')
            predictions = model.predict(X_test)
        
        logger.debug(f"Generated {len(predictions)} predictions")
        console.print(f"✓ Generated predictions", style="green")
        
        # Step 4: Create submission
        if 'id' in test_df.columns:
            ids = test_df['id'].values
        else:
            logger.warning("No 'id' column found, using sequential IDs")
            console.print("[yellow]Warning: No 'id' column, using sequential IDs[/yellow]")
            ids = np.arange(len(test_df))
        
        # Determine output path
        if output is None:
            output_dir = Path('results/predictions')
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f'task_{task}_submission.csv'
        
        logger.info(f"Creating submission file: {output}")
        console.print(f"\n[bold]Creating submission: {output}[/bold]")
        
        create_submission_from_arrays(predictions, ids, str(output))
        
        # Display submission info
        submission_df = pd.read_csv(output)
        
        info_table = Table(title="Submission Info", show_header=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        info_table.add_row("Total samples", f"{len(submission_df):,}")
        info_table.add_row("Unique labels", f"{submission_df['label'].nunique()}")
        info_table.add_row("Output file", str(output))
        
        console.print(info_table)
        
        # Label distribution
        console.print("\n[bold]Label Distribution:[/bold]")
        label_counts = submission_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            pct = (count / len(submission_df)) * 100
            console.print(f"  Label {label}: {count:,} ({pct:.1f}%)")
        
        console.rule("[bold green]Submission Ready!")
        console.print(f"\n🚀 Ready to upload to Kaggle: {output}", style="bold green")
        logger.info("Submission pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Submission pipeline failed: {e}")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def validate(
    submission_file: Path = typer.Argument(..., help="Path to submission CSV"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """
    Validate a submission file format
    
    Examples:
        python pipeline.py validate results/predictions/submission.csv
    """
    setup_logging(debug)
    
    console.rule("[bold blue]Validating Submission")
    logger.info(f"Validating submission: {submission_file}")
    
    try:
        if not submission_file.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {submission_file}")
            raise typer.Exit(code=1)
        
        df = pd.read_csv(submission_file)
        
        errors = []
        warnings = []
        
        # Check required columns
        if 'id' not in df.columns:
            errors.append("Missing 'id' column")
        if 'label' not in df.columns:
            errors.append("Missing 'label' column")
        
        # Check for extra columns
        extra_cols = set(df.columns) - {'id', 'label'}
        if extra_cols:
            warnings.append(f"Extra columns found: {extra_cols}")
        
        # Check IDs
        if 'id' in df.columns:
            if not df['id'].is_unique:
                errors.append("IDs are not unique")
            if df['id'].isna().any():
                errors.append("Missing IDs found")
        
        # Check labels
        if 'label' in df.columns:
            if df['label'].isna().any():
                errors.append("Missing labels found")
            if df['label'].dtype not in [np.int32, np.int64]:
                errors.append(f"Labels must be integers, got {df['label'].dtype}")
        
        # Display results
        if errors:
            console.print("\n[bold red]❌ Validation Failed[/bold red]")
            for error in errors:
                console.print(f"  • {error}", style="red")
            raise typer.Exit(code=1)
        
        if warnings:
            console.print("\n[bold yellow]⚠️  Warnings[/bold yellow]")
            for warning in warnings:
                console.print(f"  • {warning}", style="yellow")
        
        # Summary
        summary_table = Table(title="Validation Summary", show_header=True)
        summary_table.add_column("Check", style="cyan")
        summary_table.add_column("Status", style="green")
        
        summary_table.add_row("Required columns", "✓ Present")
        summary_table.add_row("Unique IDs", "✓ Valid")
        summary_table.add_row("No missing values", "✓ Valid")
        summary_table.add_row("Correct data types", "✓ Valid")
        
        console.print(summary_table)
        
        # Statistics
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Total samples: {len(df):,}")
        console.print(f"  Unique labels: {df['label'].nunique()}")
        console.print(f"  Label range: [{df['label'].min()}, {df['label'].max()}]")
        
        console.print("\n[bold green]✓ Submission format is valid![/bold green]")
        logger.info("Validation successful")
        
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info(
    task: str = typer.Option(..., "--task", "-t", help="Task identifier: A, B, or C"),
) -> None:
    """
    Display information about a task
    
    Examples:
        python pipeline.py info --task A
    """
    console.rule(f"[bold blue]Task {task} Information")
    
    try:
        loader = TaskDataLoader(task)
        
        # Training set info
        train_df = loader.load_split('train')
        stats = loader.get_statistics(train_df)
        
        # Display task info
        task_info = {
            'A': {
                'name': 'Binary Classification',
                'question': 'Human or AI-generated?',
                'classes': 2,
                'expected_baseline': '50-60%',
                'expected_competitive': '90%+',
            },
            'B': {
                'name': 'Multi-Class Authorship',
                'question': 'Which AI model or human?',
                'classes': 11,
                'expected_baseline': '35-45%',
                'expected_competitive': '85%+',
            },
            'C': {
                'name': 'Hybrid Detection',
                'question': 'Human, AI, Hybrid, or Adversarial?',
                'classes': 4,
                'expected_baseline': '30-40%',
                'expected_competitive': '80%+',
            },
        }
        
        info = task_info.get(task, {})
        
        info_table = Table(title=f"Task {task}: {info.get('name', 'Unknown')}", show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        info_table.add_row("Question", info.get('question', 'N/A'))
        info_table.add_row("Number of classes", str(info.get('classes', 'N/A')))
        info_table.add_row("Training samples", f"{stats['total_samples']:,}")
        info_table.add_row("Expected baseline F1", info.get('expected_baseline', 'N/A'))
        info_table.add_row("Expected competitive F1", info.get('expected_competitive', 'N/A'))
        
        console.print(info_table)
        
        # Label distribution
        if 'label_distribution' in stats:
            console.print("\n[bold]Label Distribution:[/bold]")
            for label, count in stats['label_distribution'].items():
                pct = (count / stats['total_samples']) * 100
                console.print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Train baseline: [cyan]python pipeline.py train --task {task}[/cyan]")
        console.print(f"  2. Create submission: [cyan]python pipeline.py submit --task {task} --test-data data/test_{task}.parquet[/cyan]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    app()
