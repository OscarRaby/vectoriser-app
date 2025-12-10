import json

class GUITheme:
    def __init__(self, theme_file='gui_theme.json'):
        try:
            with open(theme_file, 'r') as f:
                self.theme = json.load(f)
        except Exception:
            self.theme = {}
        self._set_defaults()

    def _set_defaults(self):
        defaults = {
            "borderless": False,
            "background": "#181828",
            "font": "Segoe UI",
            "font_size": 12,
            "input_bg": "#232336",
            "input_fg": "#EEEEEE",
            "label_fg": "#CCCCCC",
            "button_bg": "#30304A",
            "button_fg": "#EEEEEE",
            "highlight": "#5566FF"
        }
        for k, v in defaults.items():
            if k not in self.theme:
                self.theme[k] = v

    def font(self, weight="normal"):
        return (self.theme["font"], self.theme["font_size"], weight)
    def __getitem__(self, k):
        return self.theme[k]
