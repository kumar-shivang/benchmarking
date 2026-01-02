"""
Logging configuration for benchmarking system.
Creates timestamped log files for each run with detailed logging of all operations.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(name="benchmarking", log_dir="logs", level=logging.INFO):
    """
    Setup logger with both console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler - Detailed logging with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB per file
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)

    # Add only file handler (console logging disabled)
    logger.addHandler(file_handler)

    # Log the start
    logger.info("=" * 80)
    logger.info(f"Logging initialized - Log file: {log_path}")
    logger.info("=" * 80)

    return logger


def get_logger(name="benchmarking"):
    """
    Get existing logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_exception(logger, exc, context=""):
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        exc: Exception object
        context: Additional context information
    """
    if context:
        logger.error(f"Exception in {context}: {str(exc)}", exc_info=True)
    else:
        logger.error(f"Exception: {str(exc)}", exc_info=True)


def log_api_call(logger, model, operation, success=True, details=""):
    """
    Log API call details.

    Args:
        logger: Logger instance
        model: Model name
        operation: Operation description
        success: Whether operation succeeded
        details: Additional details
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"API Call [{status}] - Model: {model}, Operation: {operation}"
    if details:
        message += f", Details: {details}"

    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_database_operation(logger, operation, success=True, details=""):
    """
    Log database operation.

    Args:
        logger: Logger instance
        operation: Operation description
        success: Whether operation succeeded
        details: Additional details
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"Database [{status}] - Operation: {operation}"
    if details:
        message += f", Details: {details}"

    if success:
        logger.debug(message)
    else:
        logger.error(message)


def log_cost_summary(logger, generation_cost, evaluation_cost, total_questions):
    """
    Log cost summary.

    Args:
        logger: Logger instance
        generation_cost: Total generation cost
        evaluation_cost: Total evaluation cost
        total_questions: Number of questions processed
    """
    logger.info("=" * 80)
    logger.info("COST SUMMARY")
    logger.info(f"  Generation Cost:  ${generation_cost:.4f}")
    logger.info(f"  Evaluation Cost:  ${evaluation_cost:.4f}")
    logger.info(f"  Total Cost:       ${generation_cost + evaluation_cost:.4f}")
    logger.info(f"  Questions:        {total_questions}")
    logger.info(
        f"  Cost per Question: ${(generation_cost + evaluation_cost) / max(total_questions, 1):.4f}"
    )
    logger.info("=" * 80)


def log_run_summary(logger, run_id, student_model, evaluators, summary_data):
    """
    Log run summary details.

    Args:
        logger: Logger instance
        run_id: Run ID
        student_model: Student model name
        evaluators: List of evaluator model names
        summary_data: Summary data dictionary or dataframe
    """
    logger.info("=" * 80)
    logger.info("RUN SUMMARY")
    logger.info(f"  Run ID:         {run_id}")
    logger.info(f"  Student Model:  {student_model}")
    logger.info(f"  Evaluators:     {', '.join(evaluators)}")
    logger.info("-" * 80)
    logger.info(f"  Summary Data:\n{summary_data}")
    logger.info("=" * 80)
