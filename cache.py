import json
import os

class Cache:
    def __init__(self, file_path="data.json"):
        """
        Initializes the cache from a JSON file.
        If the file doesn't exist or is invalid, it starts with an empty cache.
        """
        self.file_path = file_path
        self.cache = {}

        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as file:
                    self.cache = json.load(file)
            except (json.JSONDecodeError, IOError):
                print("Warning: Could not load cache. Starting with an empty cache.")

    def check_cache(self, key, solve):
        """
        Checks if a key exists in the cache. If not, calculates the value using `solve`,
        adds it to the cache, and persists the cache to the file.

        Args:
            key (str): The key to look up in the cache.
            solve (function): A function that computes the value if the key is not in the cache.

        Returns:
            The value associated with the key.
        """
        if key in self.cache:
            return self.cache[key]

        # Calculate the value using the solve function
        value = solve()
        self.cache[key] = value
        self._persist_cache()
        return value

    def _persist_cache(self):
        """
        Persists the current cache to the JSON file.
        """
        try:
            with open(self.file_path, "w") as file:
                json.dump(self.cache, file, indent=4)
        except IOError as e:
            print(f"Error: Could not save cache to file: {e}")
