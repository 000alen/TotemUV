from kivy.app import App
from kivy.uix.label import Label


FUTURA_BOLD = "font/Futura Bold.tf"


class InitialScreen(App):
    def build(self):
        return Label(
            text="THE BORING TEAM!",
            font_name=FUTURA_BOLD
        )


class CaptureScreen(App):
    pass


class AnalysisScreen(App):
    pass
