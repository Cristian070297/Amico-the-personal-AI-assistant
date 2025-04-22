import time
import psutil
from typing import List, Dict, Any, Optional
import os  
# Import classes from other files within the project
from config import ConfigLoader
from database import DatabaseManager
from services import OpenAIService, ExternalApiService
from memory import MemoryRetriever
from speech import SpeechInterface
from logger import PerformanceLogger

class AmicoApp:
    """The main application orchestrator for Amico."""
    SHORT_TERM_MEMORY_LIMIT = 5 # Number of exchanges (user + assistant)

    def __init__(self):
        """Initializes all components of the Amico application."""
        self.config = ConfigLoader()
        # Use a distinct log file name for the refactored version
        self.performance_logger = PerformanceLogger(log_file='performance_log_refactored.csv')
        self.process = psutil.Process(os.getpid()) # For resource monitoring

        # Validate Keys early
        missing_keys = self.config.validate_keys()
        if missing_keys:
            print("\n*** WARNING: Missing environment variables: ***")
            print(f"Missing: {', '.join(missing_keys)}")
            print("Functionality may be limited. Ensure these are in your .env file.")
            # Decide if exit is necessary based on critical keys
            if "OPENAI_API_KEY" in missing_keys or "fine_tuned_model_id" in missing_keys:
                 critical_error = "Essential OpenAI key or fine-tuned model ID missing. Cannot proceed."
                 print(critical_error)
                 # Attempt to log fatal error before exiting
                 self.performance_logger.log('Initialization Error', 0, self.process.memory_info().rss / (1024*1024))
                 self.performance_logger.close() # Close log before exit
                 raise SystemExit(critical_error) # Exit cleanly

        # Initialize Services (Proceed only if critical keys are present)
        try:
            self.db_manager = DatabaseManager(self.config.db_path)
            self.openai_service = OpenAIService(
                self.config.openai_api_key,
                self.config.embedding_model,
                self.config.fine_tuned_model_id
            )
            self.memory_retriever = MemoryRetriever(self.db_manager, self.openai_service)
            self.external_api_service = ExternalApiService(
                self.config.gnews_api_key,
                self.config.fmp_api_key,
                self.config.weather_api_key
            )
            self.speech_interface = SpeechInterface()
        except Exception as e:
            init_error = f"Failed to initialize core services: {e}"
            print(init_error)
            self.performance_logger.log('Initialization Error', 0, self.process.memory_info().rss / (1024*1024))
            self.performance_logger.close()
            raise SystemExit(init_error)

        # Application State
        self.user_profile: Dict[str, str] = {}
        self.short_term_memory: List[Dict[str, str]] = []

    def _initialize_app(self):
        """Initializes database and loads user profile."""
        try:
            self.db_manager.initialize()
            print("Loading user profile...")
            self.user_profile = self.db_manager.fetch_user_profile()
        except Exception as e:
             # Log error but potentially allow app to continue with default/empty profile
             print(f"Error during app initialization (DB/Profile): {e}")
             self.performance_logger.log('Initialization Error (DB/Profile)', 0, self.process.memory_info().rss / (1024*1024))
             # Depending on severity, could raise SystemExit here too

    def _update_short_term_memory(self, query: str, response: str):
        """Adds the latest interaction to short-term memory."""
        if not isinstance(self.short_term_memory, list):
             self.short_term_memory = [] # Ensure it's a list
        self.short_term_memory.append({"role": "user", "content": query})
        self.short_term_memory.append({"role": "assistant", "content": response})
        # Maintain size limit (remove oldest exchange)
        if len(self.short_term_memory) > self.SHORT_TERM_MEMORY_LIMIT * 2:
            self.short_term_memory = self.short_term_memory[2:] # Remove the first two items (one user, one assistant)

    def _build_short_term_context(self) -> str:
        """Builds a string context from short-term memory."""
        if not isinstance(self.short_term_memory, list): return ""
        return "\n".join([f"{mem.get('role', 'unknown')}: {mem.get('content', '')}" for mem in self.short_term_memory])

    def _narrate_news(self, news_articles: List[Dict[str, Any]], count: int = 5):
        """Narrates a summary of news articles using the SpeechInterface."""
        if not news_articles: # Check if list is None or empty
             self.speech_interface.speak("I couldn't fetch any news articles at the moment.")
             return

        valid_articles = [a for a in news_articles if isinstance(a, dict)]
        num_to_narrate = min(count, len(valid_articles))

        if num_to_narrate == 0:
            self.speech_interface.speak("I couldn't find any valid news articles to report.")
            return

        self.speech_interface.speak(f"Here are the top {num_to_narrate} news headlines.")
        for idx, article in enumerate(valid_articles[:num_to_narrate]):
            headline = article.get('title', 'No title available')
            # Adjust source extraction based on potential GNews structure
            source = article.get('source', {}).get('name', 'Unknown source')
            self.speech_interface.speak(f"Headline {idx + 1} from {source}: {headline}.")
        self.speech_interface.speak("That's the news summary.")


    def _handle_profile_query(self, command: str) -> Optional[str]:
        """Checks if the command is a direct query about the user profile."""
        user_name = self.user_profile.get('user_name', 'User') # Default to 'User' if not set
        response = None

        # Define phrases and corresponding profile keys
        profile_key_map = {
            "my age": "age",
            "my nationality": "nationality",
            "where am i from": "nationality",
            "where do i live": "preferred_location",
            "my location": "preferred_location",
            "favorite color": "favorite_color",
            "health condition": "health_condition",
            "headaches": "health_condition" # Example specific health query
        }

        matched_key = None
        # Find the first matching phrase in the command
        for phrase, key in profile_key_map.items():
             # Use 'in' for simple substring matching
             if phrase in command:
                 matched_key = key
                 break # Use the first match found

        if matched_key:
             value = self.user_profile.get(matched_key)
             profile_field_name = matched_key.replace('_', ' ') # Make it readable

             if value:
                 # Generate response based on key
                 if matched_key == 'age': response = f"According to your profile, you are {value} years old."
                 elif matched_key == 'nationality': response = f"Your profile says your nationality is {value}."
                 elif matched_key == 'preferred_location': response = f"Your preferred location is set to {value}."
                 elif matched_key == 'favorite_color': response = f"Your profile indicates your favorite color is {value}."
                 elif matched_key == 'health_condition': response = f"Your profile notes: {value}."
                 else: response = f"Your profile has '{profile_field_name}' set to '{value}'." # Generic fallback
             else:
                 response = f"I don't have your {profile_field_name} stored in the profile."

             # Add user name personalization if not the generic 'User'
             if user_name and user_name != 'User':
                 # Append name carefully, check punctuation
                 if response.endswith('.'):
                     response = response[:-1] + f", {user_name}."
                 else:
                     response += f", {user_name}."
        # Return the generated response string or None if no profile phrase matched
        return response


    def _handle_llm_query(self, query: str) -> str:
        """Handles general queries using context retrieval and the LLM."""
        # --- Context Retrieval (Timed and Logged) ---
        start_context_time = time.perf_counter()
        start_mem_rss = self.process.memory_info().rss / (1024 * 1024)

        # Retrieve context string, number fetched, number scored
        long_term_context, fetched, scored = self.memory_retriever.retrieve_relevant_context(
            query_text=query,
            fetch_limit=100, # How many raw memories to fetch initially
            context_limit=5  # How many top memories to include in final context
        )

        end_context_time = time.perf_counter()
        context_retrieval_duration_ms = (end_context_time - start_context_time) * 1000
        end_mem_rss = self.process.memory_info().rss / (1024 * 1024)

        # Log context retrieval performance
        self.performance_logger.log(
            'ContextRetrieval', context_retrieval_duration_ms, end_mem_rss, len(query), fetched, scored
        )
        print(f"DEBUG: Context retrieval: {context_retrieval_duration_ms:.2f} ms. Fetched={fetched}, Scored={scored}")

        # --- Build Contexts for LLM ---
        short_term_context = self._build_short_term_context()

        # Build profile context string only if profile has data
        profile_context_parts = []
        if self.user_profile:
             profile_context_parts.append("User Profile Summary:")
             for key, value in self.user_profile.items():
                 formatted_key = key.replace('_', ' ').title()
                 profile_context_parts.append(f"- {formatted_key}: {value}")
        profile_context_str = "\n".join(profile_context_parts) if profile_context_parts else "No user profile data available."


        # Combine all context pieces
        full_context = (
            f"{profile_context_str}\n\n"
            f"Recent conversation (Short-Term Memory):\n{short_term_context if short_term_context else 'None'}\n\n"
            f"Relevant past interactions (Long-Term Memory):\n{long_term_context}"
        )

        # --- Prepare LLM Request ---
        system_prompt = "You are Amico, a helpful and conversational AI assistant with memory. Be concise unless asked for detail. Use the provided User Profile Summary and conversation context when relevant."
        user_name = self.user_profile.get('user_name')
        if user_name and user_name != 'User': # Add personalization if name is available
            system_prompt += f" Address the user as {user_name} when appropriate."

        # Structure messages for the chat API
        messages = [
            {"role": "system", "content": system_prompt},
            # Provide context clearly separated from the current query
            {"role": "user", "content": f"CONTEXT:\n{full_context}\n\nCURRENT QUERY:\n{query}"}
        ]

        # --- LLM Call (Timed and Logged) ---
        llm_start_time = time.perf_counter()
        # Use the dedicated method which handles API calls and basic error reporting
        response_text = self.openai_service.get_chat_completion(messages)
        llm_end_time = time.perf_counter()
        llm_duration_ms = (llm_end_time - llm_start_time) * 1000
        llm_mem_rss = self.process.memory_info().rss / (1024 * 1024)

        # Determine log operation based on whether the response indicates an error
        log_operation = 'LLM_API_Call_FineTuned'
        # Check common error indicators in the returned string
        if response_text is None or any(err in response_text.lower() for err in ["error", "sorry, i encountered"]):
             log_operation = 'LLM_API_Error_FineTuned'

        self.performance_logger.log(log_operation, llm_duration_ms, llm_mem_rss, len(query))
        print(f"DEBUG: LLM call: {llm_duration_ms:.2f} ms")

        # --- Save interaction to memory *only if successful* ---
        if log_operation == 'LLM_API_Call_FineTuned':
            try:
                # Get embeddings for both query and response to save them
                query_embedding = self.openai_service.get_embedding(query)
                response_embedding = self.openai_service.get_embedding(response_text)
                # Save user query and assistant response to long-term memory (database)
                self.db_manager.save_to_memory("user", query, query_embedding)
                self.db_manager.save_to_memory("assistant", response_text, response_embedding)
                # Update short-term memory as well
                self._update_short_term_memory(query, response_text)
            except Exception as e:
                 print(f"Error saving interaction to memory after successful LLM call: {e}")
                 # Log this potential secondary failure?
        else:
             print("Skipping saving interaction to memory due to potential error in LLM response.")

        return response_text if response_text is not None else "An unexpected issue occurred."

    def _extract_stock_symbol(self, command: str) -> Optional[str]:
         """Tries to extract a stock symbol from the command string."""
         words = command.split()
         # 1. Check after keywords
         keywords = ["stock", "of", "shares", "for", "price"]
         for i, word in enumerate(words):
             if word in keywords and i + 1 < len(words):
                 potential_symbol = words[i+1]
                 # Basic check: often symbols are short and alphabetic
                 if 1 < len(potential_symbol) < 6 and potential_symbol.isalpha():
                      return potential_symbol.upper()

         # 2. Look for an all-caps word (common for symbols)
         for word in words:
             # Check if word is all uppercase, alphabetic, and typical symbol length
             if word.isupper() and word.isalpha() and 1 < len(word) < 6:
                 return word

         # 3. Fallback: If no clear symbol found
         return None

    def run(self):
        """Starts the main application loop."""
        self._initialize_app() # Initialize DB and load profile

        user_name = self.user_profile.get('user_name', None)
        greeting = f"Hi {user_name}, I am Amico." if user_name and user_name != 'User' else "Hi, I am Amico."
        self.speech_interface.speak(f"{greeting} How can I assist you today?")

        try:
            while True:
                # --- Listen for Command ---
                interaction_start_time = time.perf_counter()
                command = self.speech_interface.listen() # Blocks here
                command_received_time = time.perf_counter() # Time right after listen() returns

                # --- Handle Empty Input ---
                if not command:
                    print("No command received (timeout or error).")
                    time.sleep(0.1) # Small delay to prevent tight loop on continuous errors
                    continue

                # --- Process Command ---
                processing_start_time = time.perf_counter()
                normalized_command = command.strip().lower()
                response_text = None
                operation_type = 'Unknown_Command' # Default log type

                # 1. Check for Exit Command
                if any(word in normalized_command.split() for word in ["exit", "quit", "goodbye", "stop listening", "bye"]):
                    response_text = "Goodbye! Have a great day."
                    operation_type = 'Command_Exit'
                    self.speech_interface.speak(response_text) # Speak before breaking
                    break # Exit the main loop

                # 2. Check for Stop Speaking Command
                elif "stop speaking" in normalized_command:
                    self.speech_interface.stop_speaking()
                    response_text = "Okay, I've stopped speaking."
                    operation_type = 'Command_Stop_Speaking'
                    # No memory saving needed for this command
                    # No further TTS needed

                # 3. Check for Profile Queries
                elif profile_response := self._handle_profile_query(normalized_command):
                     response_text = profile_response
                     operation_type = 'Profile_Query'
                     # Profile queries don't modify memory (currently)
                     # Needs TTS

                # 4. Check for Weather Command
                elif "weather" in normalized_command:
                    operation_type = 'Command_Weather'
                    location = None
                    # Try extracting location after specific phrases
                    parts = normalized_command.split("weather in ")
                    if len(parts) > 1: location = parts[1].strip().rstrip('?.!')
                    else:
                        parts = normalized_command.split("weather for ")
                        if len(parts) > 1: location = parts[1].strip().rstrip('?.!')

                    # If not found in command, use profile or ask
                    if not location:
                        location = self.user_profile.get('preferred_location')
                        if location:
                            self.speech_interface.speak(f"Getting weather for your preferred location: {location}")
                        else:
                            self.speech_interface.speak("Which location's weather are you interested in?")
                            loc_cmd = self.speech_interface.listen() # Listen again for location
                            location = loc_cmd.strip() if loc_cmd else None

                    # Fetch weather if location is determined
                    if location:
                        response_text = self.external_api_service.get_weather(location)
                    else:
                        response_text = "Sorry, I couldn't determine a location for the weather."
                    # Needs TTS

                # 5. Check for News Command
                elif "news" in normalized_command:
                    operation_type = 'Command_News'
                    news_articles = self.external_api_service.get_general_news()
                    # Narration happens within this call, including TTS
                    self._narrate_news(news_articles)
                    # Set response_text for logging/memory, but *don't* speak it again here
                    response_text = f"Provided news summary ({len(news_articles)} articles fetched)."
                    # Needs memory saving

                # 6. Check for Stock Command
                elif any(word in normalized_command for word in ["stock", "price of", "shares", "financial"]):
                    operation_type = 'Command_Stock'
                    symbol = self._extract_stock_symbol(normalized_command)

                    # If symbol not extracted, ask user
                    if not symbol:
                        self.speech_interface.speak("Which stock symbol are you interested in?")
                        symbol_cmd = self.speech_interface.listen()
                        if symbol_cmd:
                            # Try extracting again from the new input
                             symbol = self._extract_stock_symbol(symbol_cmd.lower())

                    # Fetch data if symbol is determined
                    if symbol:
                        response_text = self.external_api_service.get_stock_data(symbol)
                    else:
                        response_text = "Sorry, I couldn't identify a stock symbol."
                    # Needs TTS

                # 7. Default to General LLM Query
                else:
                    operation_type = 'E2E_Latency_General'
                    # This method handles context retrieval, LLM call, and internal memory saving if successful
                    response_text = self._handle_llm_query(normalized_command)
                    # Needs TTS (handled below)

                # --- Common Post-Processing Block ---
                processing_end_time = time.perf_counter() # Time after specific handler finishes

                # Speak the response if one was generated and not already spoken (like news)
                if response_text and operation_type != 'Command_News' and operation_type != 'Command_Stop_Speaking':
                    self.speech_interface.speak(response_text)

                # Save interaction to memory for commands that didn't use the LLM handler
                # (LLM handler saves internally only on success)
                # Exclude Exit, Stop Speaking, and Profile queries from saving.
                if response_text and operation_type not in ['E2E_Latency_General', 'Command_Exit', 'Command_Stop_Speaking', 'Profile_Query']:
                     try:
                         # Get embeddings for saving
                         query_embedding = self.openai_service.get_embedding(command)
                         response_embedding = self.openai_service.get_embedding(response_text)
                         # Save to DB
                         self.db_manager.save_to_memory("user", command, query_embedding)
                         self.db_manager.save_to_memory("assistant", response_text, response_embedding)
                         # Update STM
                         self._update_short_term_memory(command, response_text)
                     except Exception as e:
                          print(f"Error saving non-LLM interaction to memory: {e}")


                # --- Log E2E Latency ---
                tts_finished_time = time.perf_counter() # Time after speak() returns (or immediately if no speak needed)
                # Calculate latency from the moment the command was received until processing+speaking finished
                total_latency_ms = (tts_finished_time - command_received_time) * 1000
                current_mem_rss = self.process.memory_info().rss / (1024 * 1024)

                print(f"--- Performance Metrics ({operation_type}) ---")
                print(f"Total E2E Latency (End Listen -> End Speak/Process): {total_latency_ms:.2f} ms")
                print(f"Memory Usage (RSS) after processing: {current_mem_rss:.2f} MB")
                print(f"--------------------------------------------------")

                # Log E2E latency for the completed operation
                # Fetched/Scored counts are None here as they are logged internally by _handle_llm_query if relevant
                self.performance_logger.log(operation_type, total_latency_ms, current_mem_rss, len(command))

        except KeyboardInterrupt:
            print("\nExiting Amico loop due to Ctrl+C.")
            self.speech_interface.speak("Shutting down.")
        except SystemExit as e: # Catch SystemExit raised during init
             print(f"Exiting due to initialization error: {e}")
        except Exception as e: # Catch unexpected errors during the loop
             print(f"\n--- UNEXPECTED ERROR IN MAIN LOOP ---")
             print(f"Error: {e}")
             import traceback
             traceback.print_exc() # Print stack trace for debugging
             self.performance_logger.log('Runtime Error', 0, self.process.memory_info().rss / (1024*1024))
             self.speech_interface.speak("An unexpected error occurred. Please check the console.")
        finally:
            # Ensure cleanup happens regardless of how the loop exits
            print("Performing final cleanup...")
            self.performance_logger.close()