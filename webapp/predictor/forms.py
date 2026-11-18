from django import forms

from .ml import CATEGORIES


def _nutrition_field(label, unit):
    return forms.FloatField(
        label=label, required=False, min_value=0,
        help_text=f"{unit}. Leave blank to use the category median.",
        widget=forms.NumberInput(attrs={"step": "any", "placeholder": "optional"}))


class RecipeForm(forms.Form):
    category = forms.ChoiceField(choices=[(c, c) for c in CATEGORIES])
    servings = forms.IntegerField(min_value=1, max_value=50, initial=4)
    calories = _nutrition_field("Calories", "kcal")
    carbohydrate = _nutrition_field("Carbohydrate", "grams")
    sugar = _nutrition_field("Sugar", "grams")
    protein = _nutrition_field("Protein", "grams")


class BatchUploadForm(forms.Form):
    file = forms.FileField(
        label="Recipe CSV",
        help_text="Required columns: category, servings. Optional: calories, "
                  "carbohydrate, sugar, protein, and any ID columns (kept as-is).")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a .csv file.")
        return f


class RankingFilterForm(forms.Form):
    category = forms.ChoiceField(
        required=False, choices=[("", "All categories")] + [(c, c) for c in CATEGORIES])
    top = forms.TypedChoiceField(
        coerce=int, initial=30, choices=[(n, f"Top {n}") for n in (10, 30, 50, 100)],
        required=False)
    above_threshold = forms.BooleanField(required=False, initial=True,
                                         label="Only recommended (≥ threshold)")
