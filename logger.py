# logger.py
import csv
import os
from datetime import datetime
from typing import Optional

class PerformanceLogger:
    """Handles logging performance metrics to a CSV file."""
    def __init__(self, log_file='performance_log.csv'):
        self.log_file = log_file
        self._log_handle = None
        self._log_writer = None
        self._is_initialized = False
        self._initialize_log()

    def _initialize_log(self):
        """Initializes the log file and writer."""
        try:
            log_exists = os.path.exists(self.log_file)
            # Use 'a+' mode, create file if not exists, append otherwise
            # Specify encoding to avoid potential issues
            self._log_handle = open(self.log_file, 'a+', newline='', encoding='utf-8')
            self._log_writer = csv.writer(self._log_handle)
            # Write header only if the file was just created or is empty
            if not log_exists or os.path.getsize(self.log_file) == 0:
                self._log_writer.writerow([
                    'Timestamp', 'Operation', 'Duration_ms', 'Memory_RSS_MB',
                    'Query_Length', 'Memories_Fetched', 'Memories_Scored'
                ])
                self._log_handle.flush() # Ensure header is written immediately
            self._is_initialized = True
        except IOError as e:
             print(f"Error initializing performance logger: {e}")
             self._is_initialized = False
             if self._log_handle: # Try to close if opened partially
                  self._log_handle.close()
             self._log_handle = None
             self._log_writer = None


    def log(self, operation: str, duration_ms: float, memory_rss_mb: float,
            query_length: Optional[int] = None,
            memories_fetched: Optional[int] = None,
            memories_scored: Optional[int] = None):
        """Logs a performance entry."""
        if not self._is_initialized or not self._log_writer or not self._log_handle:
            print("Warning: Performance logger not initialized or failed to initialize. Skipping log entry.")
            return

        try:
            self._log_writer.writerow([
                datetime.now().isoformat(), operation, round(duration_ms, 2), round(memory_rss_mb, 2),
                query_length if query_length is not None else '',
                memories_fetched if memories_fetched is not None else '',
                memories_scored if memories_scored is not None else ''
            ])
            self._log_handle.flush() # Ensure data is written immediately
        except Exception as e:
            print(f"Error writing to performance log: {e}")


    def close(self):
        """Closes the log file handle."""
        if self._log_handle and not self._log_handle.closed:
            try:
                print("Closing performance log file.")
                self._log_handle.close()
                self._is_initialized = False
            except Exception as e:
                 print(f"Error closing performance log file: {e}")