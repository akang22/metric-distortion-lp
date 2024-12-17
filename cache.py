import json
import os
from multiprocessing import Lock

def init_lock(l):
    global lock
    lock = l

class Cache:
    def __init__(self, file_path="data.json"):
        """
        Initializes the cache from a JSON file.
        If the file doesn't exist or is invalid, it starts with an empty cache.
        """
        self.file_path = file_path
        self.cache = {}
        init_lock(Lock())
        try:
            lock.acquire()
            self.load_file()
        finally:
            lock.release()


    def load_file(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as file:
                file.write("{}")
        try:
            with open(self.file_path, "r") as file:
                self.cache = { **self.cache, **json.load(file)}
        except (json.JSONDecodeError, IOError) as e:
            print("Error: Could not load cache.")
            raise e

    def check_cache(self, key, solve, use_cache=True, persist_cache=True):
        """
        Checks if a key exists in the cache. If not, calculates the value using `solve`,
        adds it to the cache, and persists the cache to the file.

        Args:
            key (str): The key to look up in the cache.
            solve (function): A function that computes the value if the key is not in the cache.

        Returns:
            The value associated with the key.
        """
        global lock
        if key in self.cache and use_cache:
            return self.cache[key]

        # Calculate the value using the solve function
        value = solve()
        self.cache[key] = value
        if persist_cache:
            try:
                lock.acquire()
                self.load_file()
                self._persist_cache()
            finally:
                lock.release()
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
