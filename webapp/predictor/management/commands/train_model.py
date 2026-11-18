from django.core.management.base import BaseCommand

from predictor import ml


class Command(BaseCommand):
    help = "Train the recipe traffic model and save it to settings.MODEL_PATH."

    def handle(self, *args, **options):
        bundle = ml.train()
        ml.save(bundle)
        m = bundle["metrics"]
        self.stdout.write(self.style.SUCCESS(
            f"Model saved. C={m['best_C']}, threshold={bundle['threshold']:.3f}, "
            f"test precision={m['business']['precision']:.3f}, "
            f"test ROC-AUC={m['test_roc_auc']:.3f}"))
