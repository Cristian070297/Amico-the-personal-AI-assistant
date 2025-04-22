import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """Handles all interactions with the SQLite database."""
    def __init__(self, db_path: str, embedding_dim: int = 1536):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.default_embedding_vector = [0.0] * self.embedding_dim

    def initialize(self):
        """Initializes the database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT
                )
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """)
                # Add default profile data if table is new/empty
                cursor.execute("INSERT OR IGNORE INTO user_profile (key, value) VALUES (?, ?)", ('user_name', 'User'))
                cursor.execute("INSERT OR IGNORE INTO user_profile (key, value) VALUES (?, ?)", ('preferred_location', 'London'))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error during initialization: {e}")

    def _convert_embedding_to_str(self, embedding_vector: List[float]) -> str:
        """Converts embedding vector list to comma-separated string."""
        return ",".join(map(str, embedding_vector))

    def _convert_str_to_embedding(self, embedding_str: Optional[str]) -> List[float]:
        """Converts comma-separated string back to embedding vector list."""
        if not embedding_str:
            return self.default_embedding_vector[:] # Return copy
        try:
            embedding = [float(x) for x in embedding_str.split(",")]
            # Validate dimension
            if len(embedding) == self.embedding_dim:
                return embedding
            else:
                print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}. Returning default.")
                return self.default_embedding_vector[:] # Return copy
        except (ValueError, TypeError) as e:
            print(f"Error parsing embedding string '{embedding_str[:50]}...': {e}. Returning default.")
            return self.default_embedding_vector[:] # Return copy

    def save_to_memory(self, role: str, content: str, embedding_vector: List[float]):
        """Saves interaction content and its embedding to the database."""
        if not content:
            return
        timestamp = datetime.now().isoformat()
        embedding_str = self._convert_embedding_to_str(embedding_vector)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO memory (timestamp, role, content, embedding)
                VALUES (?, ?, ?, ?)
                """, (timestamp, role, content, embedding_str))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error saving to memory: {e}")
        except Exception as e:
            print(f"Unexpected error saving to memory: {e}")

    def fetch_recent_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches the most recent memories from the database."""
        memories = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT timestamp, role, content, embedding
                FROM memory
                ORDER BY timestamp DESC
                LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()

                for ts_str, role, content, emb_str in rows:
                    try:
                        timestamp = datetime.fromisoformat(ts_str)
                        embedding = self._convert_str_to_embedding(emb_str)
                        memories.append({
                            "timestamp": timestamp,
                            "role": role,
                            "content": content,
                            "embedding": embedding
                        })
                    except Exception as e:
                        print(f"Error processing memory row. Content: {content[:50]}... Error: {e}")
                        # Optionally append with default embedding or skip
                        memories.append({
                            "timestamp": datetime.now(), # Fallback timestamp
                            "role": role,
                            "content": content,
                            "embedding": self.default_embedding_vector[:] # Default on error
                        })
            return memories
        except sqlite3.Error as e:
            print(f"Database error fetching recent memories: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error fetching recent memories: {e}")
            return []

    def fetch_user_profile(self) -> Dict[str, str]:
        """Fetches static user profile data from the user_profile table."""
        profile_data = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if table exists first
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile';")
                if cursor.fetchone():
                    cursor.execute("SELECT key, value FROM user_profile")
                    rows = cursor.fetchall()
                    for key, value in rows:
                        profile_data[key] = value
                else:
                    print("User profile table not found in the database.")
        except sqlite3.Error as e:
            print(f"Database error fetching user profile: {e}")
        except Exception as e:
            print(f"Unexpected error fetching user profile: {e}")
        return profile_data