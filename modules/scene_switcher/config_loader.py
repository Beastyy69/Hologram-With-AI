import json
import os

class SceneConfig:
    def __init__(self, path="config/scenes.json"):
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return {"default": "space", "calm": "forest", "energetic": "city"}