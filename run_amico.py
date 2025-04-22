# run_amico.py
import sys
from app import AmicoApp # Import the main app class from app.py

if __name__ == "__main__":
    print("Starting Amico...")
    amico_instance = None
    try:
        # Create and run the application instance
        amico_instance = AmicoApp()
        amico_instance.run()

    except SystemExit as e:
        # Handle clean exits initiated by the app (e.g., missing critical keys)
        print(f"Application exited: {e}")
        sys.exit(1) # Exit with a non-zero code to indicate an issue
    except KeyboardInterrupt:
        print("\nAmico stopped by user (KeyboardInterrupt).")
        # Cleanup might have already happened in AmicoApp's finally block,
        # but ensure logger is closed if interruption happened before app.run() finished cleanup.
        if amico_instance and amico_instance.performance_logger:
             amico_instance.performance_logger.close()
        sys.exit(0) # Clean exit code for user interruption
    except Exception as e:
         # Catch any other unexpected errors during startup or runtime
         print(f"\n--- A FATAL UNEXPECTED ERROR OCCURRED ---")
         print(f"Error: {e}")
         import traceback
         traceback.print_exc()
         # Attempt to close logger if instance exists
         if amico_instance and amico_instance.performance_logger:
              amico_instance.performance_logger.log('Fatal Error', 0, 0) # Log minimal error info
              amico_instance.performance_logger.close()
         sys.exit(1) # Exit with error code
    finally:
        # This block executes even after sys.exit() in some cases, but primary cleanup
        # should be within AmicoApp's finally or handled above.
        print("Amico script finished.")
