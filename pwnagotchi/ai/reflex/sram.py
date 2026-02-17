import os
import pickle
import logging
import tempfile
import shutil
from threading import Lock

class ReflexSRAM:
    """
    Manages the persistence of the Reflex Brain (Gymnasium) state.
    
    Acts as the 'epigenetic' memory, preserving baseline tension, 
    learned reflexes, and adaptation rates across reboots.
    
    This ensures the Kaligotchi doesn't wake up as a 'Tabula Rasa' 
    but remembers the 'mood' of its last life.
    """
    def __init__(self, path="/root/brain/reflex_brain.pkl"):
        self.file_path = path
        self.log = logging.getLogger("reflex_sram")
        self.lock = Lock()
        
        # Ensure the directory exists
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def load(self):
        """
        Awakens the reflex brain from disk.
        Returns the state dictionary or None if no memory exists.
        """
        if not os.path.exists(self.file_path):
            self.log.info("ReflexSRAM: No epigenetic memory found (Tabula Rasa).")
            return None

        with self.lock:
            try:
                with open(self.file_path, 'rb') as f:
                    state = pickle.load(f)
                    self.log.info(f"ReflexSRAM: Awakened with {len(state)} synaptic states.")
                    return state
            except (EOFError, pickle.UnpicklingError) as e:
                self.log.warning(f"ReflexSRAM: Memory corruption detected ({e}). Resetting to Tabula Rasa.")
                return None
            except Exception as e:
                self.log.error(f"ReflexSRAM: Failed to load memory: {e}")
                return None

    def save(self, state):
        """
        Hibernates the reflex brain to disk.
        Uses atomic writes (write-temp-and-rename) to prevent corruption during power loss.
        """
        if not state:
            return

        with self.lock:
            try:
                directory = os.path.dirname(self.file_path)
                
                # Write to a temporary file first
                with tempfile.NamedTemporaryFile('wb', dir=directory, delete=False) as tf:
                    pickle.dump(state, tf)
                    temp_name = tf.name
                
                # Atomic rename to overwrite the old brain
                # This ensures we never have a half-written file if power cuts
                shutil.move(temp_name, self.file_path)
                
                # Optional: Sync to ensure it hits the physical SD card
                # os.sync() # Can be expensive, use sparingly
                
            except Exception as e:
                self.log.error(f"ReflexSRAM: Failed to hibernate: {e}")
                # Clean up temp file if it was left behind
                if 'temp_name' in locals() and os.path.exists(temp_name):
                    os.remove(temp_name)