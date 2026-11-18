import json

import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase
from django.urls import reverse

from . import ml


class MLTests(SimpleTestCase):
    def test_model_meets_precision_target(self):
        bundle = ml.get_bundle()
        self.assertGreaterEqual(bundle["metrics"]["business"]["precision"], ml.TARGET_PRECISION)
        self.assertGreater(bundle["threshold"], 0.5)

    def test_prepare_input_cleans_like_training(self):
        medians = ml.get_bundle()["category_medians"]
        df = pd.DataFrame([{"category": " Chicken Breast ", "servings": "4 as a snack",
                            "calories": None}])
        out = ml.prepare_input(df, medians)
        self.assertEqual(out.loc[0, "category"], "Chicken")
        self.assertEqual(out.loc[0, "servings"], 4)
        self.assertEqual(out.loc[0, "calories"], medians["Chicken"]["calories"])

    def test_category_drives_prediction(self):
        veg = ml.predict_one(category="Vegetable", servings=4)
        bev = ml.predict_one(category="Beverages", servings=4)
        self.assertEqual(veg["prediction"], "High")
        self.assertEqual(bev["prediction"], "Low")

    def test_rejects_unknown_category_and_negative_values(self):
        medians = ml.get_bundle()["category_medians"]
        with self.assertRaises(ml.InputError):
            ml.prepare_input(pd.DataFrame([{"category": "Soup", "servings": 2}]), medians)
        with self.assertRaises(ml.InputError):
            ml.prepare_input(pd.DataFrame([{"category": "Pork", "servings": 2, "sugar": -1}]),
                             medians)


class ViewTests(SimpleTestCase):
    def test_pages_render(self):
        for name in ("predict", "batch", "rankings", "insights"):
            with self.subTest(page=name):
                self.assertEqual(self.client.get(reverse(f"predictor:{name}")).status_code, 200)

    def test_predict_form(self):
        resp = self.client.post(reverse("predictor:predict"),
                                {"category": "Potato", "servings": 4, "calories": 200})
        self.assertContains(resp, "recommended for the homepage")

    def test_predict_form_validation(self):
        resp = self.client.post(reverse("predictor:predict"), {"category": "Potato", "servings": 0})
        self.assertContains(resp, "greater than or equal to 1")

    def test_batch_upload_and_download(self):
        csv = b"recipe,category,servings,calories\n1,Vegetable,4,100\n2,Beverages,1,NA\n"
        resp = self.client.post(reverse("predictor:batch"),
                                {"file": SimpleUploadedFile("r.csv", csv), "download": "1"})
        self.assertEqual(resp["Content-Type"], "text/csv")
        body = resp.content.decode()
        self.assertIn("probability_high", body)
        self.assertEqual(body.splitlines()[1].split(",")[0], "1")  # highest first

    def test_batch_reports_errors(self):
        resp = self.client.post(reverse("predictor:batch"),
                                {"file": SimpleUploadedFile("r.csv", b"category\nPork\n")})
        self.assertContains(resp, "Missing required column(s): servings")

    def test_rankings_filter(self):
        resp = self.client.get(reverse("predictor:rankings"), {"category": "Beverages", "top": 10})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(all(r["category"] == "Beverages" for r in resp.context["rows"]))

    def test_api_single_and_batch(self):
        url = reverse("predictor:api_predict")
        one = self.client.post(url, json.dumps({"category": "Pork", "servings": 6}),
                               content_type="application/json").json()
        self.assertIn(one["prediction"], ("High", "Low"))
        many = self.client.post(url, json.dumps([{"category": "Pork", "servings": 6},
                                                 {"category": "Dessert", "servings": 2}]),
                                content_type="application/json").json()
        self.assertEqual(len(many["results"]), 2)

    def test_api_errors(self):
        url = reverse("predictor:api_predict")
        self.assertEqual(self.client.post(url, "nope", content_type="application/json").status_code, 400)
        resp = self.client.post(url, json.dumps({"category": "Soup", "servings": 1}),
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(self.client.get(url).status_code, 405)
