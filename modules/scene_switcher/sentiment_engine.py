class SentimentEngine:
    def get_mood(self, text):
        text = text.lower()
        if any(word in text for word in ['happy', 'excited', 'party', 'cool']):
            return "energetic"
        elif any(word in text for word in ['relax', 'sleep', 'quiet', 'calm']):
            return "calm"
        return "default"