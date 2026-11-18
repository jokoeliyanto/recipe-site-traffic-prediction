import json

import pandas as pd
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import ml
from .forms import BatchUploadForm, RankingFilterForm, RecipeForm

MAX_BATCH_ROWS = 10_000
PREVIEW_ROWS = 200

AUTHOR = {
    "name": "Joko Eliyanto",
    "role": "Certified Data Scientist (DataCamp)",
    "linkedin": "https://www.linkedin.com/in/joko-eliyanto-23a1b6143/",
    "github": "https://github.com/jokoeliyanto",
}
CERTIFICATE = {
    "title": "Data Scientist Professional",
    "id": "DS0026305242463",
    "date": "November 10, 2026",
}


def _common_context():
    bundle = ml.get_bundle()
    return {"threshold": bundle["threshold"], "metrics": bundle["metrics"]}


def project_view(request):
    return render(request, "predictor/project.html", _common_context())


def predict_view(request):
    form = RecipeForm(request.POST or None)
    result = None
    if request.method == "POST" and form.is_valid():
        result = ml.predict_one(**form.cleaned_data)
        result["category_rate"] = next(
            s["high_rate"] for s in ml.get_bundle()["category_stats"]
            if s["category"] == form.cleaned_data["category"])
    return render(request, "predictor/predict.html",
                  {**_common_context(), "form": form, "result": result})


def batch_view(request):
    form = BatchUploadForm(request.POST or None, request.FILES or None)
    context = {**_common_context(), "form": form}
    if request.method == "POST" and form.is_valid():
        try:
            df = pd.read_csv(form.cleaned_data["file"], dtype=str, keep_default_na=False,
                             na_values=["", "NA", "NaN", "nan", "null"])
            if len(df) > MAX_BATCH_ROWS:
                raise ml.InputError(f"Files are limited to {MAX_BATCH_ROWS:,} rows.")
            scored = ml.predict(df).sort_values("probability_high", ascending=False)
        except ml.InputError as exc:
            form.add_error("file", str(exc))
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
            form.add_error("file", "Could not read the file as CSV.")
        else:
            if "download" in request.POST:
                response = HttpResponse(content_type="text/csv")
                response["Content-Disposition"] = 'attachment; filename="recipe_predictions.csv"'
                scored.to_csv(response, index=False)
                return response
            context.update(
                columns=list(scored.columns),
                rows=scored.head(PREVIEW_ROWS).itertuples(index=False),
                total=len(scored),
                n_high=int(scored["prediction"].eq("High").sum()),
                truncated=len(scored) > PREVIEW_ROWS,
            )
    return render(request, "predictor/batch.html", context)


def rankings_view(request):
    form = RankingFilterForm(request.GET or {"top": 30, "above_threshold": True})
    form.is_valid()
    opts = form.cleaned_data

    raw = pd.read_csv(settings.RECIPE_DATA_PATH, dtype={"recipe": str})
    scored = ml.predict(raw.drop(columns="high_traffic"))
    scored["actual"] = raw["high_traffic"].fillna("Low")
    if opts.get("category"):
        scored = scored[scored["category"] == opts["category"]]
    if opts.get("above_threshold"):
        scored = scored[scored["prediction"] == "High"]
    ranked = scored.sort_values("probability_high", ascending=False)
    top = ranked.head(opts.get("top") or 30)

    return render(request, "predictor/rankings.html", {
        **_common_context(),
        "form": form,
        "rows": top.to_dict("records"),
        "n_candidates": len(ranked),
        "hit_rate": top["actual"].eq("High").mean() if len(top) else None,
    })


def insights_view(request):
    bundle = ml.get_bundle()
    coefs = bundle["coefficients"]
    max_abs = max(abs(c["coef"]) for c in coefs) or 1
    return render(request, "predictor/insights.html", {
        **_common_context(),
        "category_stats": bundle["category_stats"],
        "coefficients": [{**c, "width": abs(c["coef"]) / max_abs * 100} for c in coefs],
        "target_precision": ml.TARGET_PRECISION,
    })


def certification_view(request):
    return render(request, "predictor/certification.html", {
        **_common_context(), "author": AUTHOR, "certificate": CERTIFICATE})


@csrf_exempt
@require_POST
def api_predict(request):
    """POST a recipe object or a list of them; returns probabilities and labels.

    Example: {"category": "Vegetable", "servings": 4, "calories": 120.5}
    """
    try:
        payload = json.loads(request.body or b"null")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be valid JSON."}, status=400)

    single = isinstance(payload, dict)
    records = [payload] if single else payload
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        return JsonResponse({"error": "Send a recipe object or a non-empty list of them."},
                            status=400)
    if len(records) > MAX_BATCH_ROWS:
        return JsonResponse({"error": f"At most {MAX_BATCH_ROWS:,} recipes per request."},
                            status=400)
    try:
        scored = ml.predict(pd.DataFrame(records))
    except ml.InputError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    threshold = ml.get_bundle()["threshold"]
    results = [{"probability_high": float(p), "prediction": label}
               for p, label in zip(scored["probability_high"], scored["prediction"])]
    body = {"threshold": round(threshold, 4)}
    body.update(results[0] if single else {"results": results})
    return JsonResponse(body)
