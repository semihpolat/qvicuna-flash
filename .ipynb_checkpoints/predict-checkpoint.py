from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def predict(self, message: str = Input(default="Hello")) -> str:
        return f"qVicuna Flash says: {message}! 🚀"
