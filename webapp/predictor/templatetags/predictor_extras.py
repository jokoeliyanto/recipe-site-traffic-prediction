from django import template

register = template.Library()


@register.filter
def pct(value, digits=0):
    """0.613 -> '61%'."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "–"


@register.filter
def pct_width(value):
    """Clamp a 0–1 value to a CSS width percentage."""
    try:
        return f"{min(max(float(value), 0), 1) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0%"
