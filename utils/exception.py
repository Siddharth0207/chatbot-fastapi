"""Custom exception handling module for enhanced error reporting.

Provides exception classes and utilities for capturing detailed error information
including file names, line numbers, and formatted error messages.
"""

import sys 
from utils.logger import logging


def error_message_detail(error, error_detail: sys) -> str:
    """Extract and format detailed error information from exception traceback.
    
    Args:
        error: The exception or error object.
        error_detail (sys): System module reference to access exception info via exc_info().
    
    Returns:
        str: Formatted error message containing script name, line number, and error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    """Custom exception class with detailed error tracking.
    
    Captures exception details including the file name and line number where
    the error occurred, providing comprehensive error information for debugging.
    
    Attributes:
        error_message (str): Detailed error message with file, line, and error info.
    """
    
    def __init__(self, error_message: str, error_details: sys):
        """Initialize CustomException with detailed error information.
        
        Args:
            error_message (str): The error message to include in the exception.
            error_details (sys): System module reference for traceback extraction.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self) -> str:
        """Return the detailed error message.
        
        Returns:
            str: Formatted error message with file name, line number, and error details.
        """
        return self.error_message