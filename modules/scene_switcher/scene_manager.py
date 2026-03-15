class SceneManager:
    def __init__(self, config):
        self.scenes = config
        self.current_scene = "default"

    def update_scene(self, mood):
        if mood in self.scenes:
            self.current_scene = mood
        return self.current_scene