class VoiceCommands:
    def __init__(self):
        self.triggers = {
            "set to forest": "calm",
            "go to space": "default",
            "city life": "energetic"
        }

    def parse_command(self, audio_text):
        for phrase, scene in self.triggers.items():
            if phrase in audio_text.lower():
                return scene
        return None