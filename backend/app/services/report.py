"""PDF report. Everything printed is recomputed on the server from validated input."""

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.schemas.prediction import Prediction, Profile

DISCLAIMER = (
    "This is a statistical estimate for groups of similar people, not a forecast for you and not medical "
    "advice. Factor effects come from US survey data (NHANES) applied to Indian death rates (SRS life "
    "tables 2020-24). They show associations, not proof of cause."
)


def label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def build_report(profile: Profile, result: Prediction) -> bytes:
    styles = getSampleStyleSheet()
    estimate = result.estimate

    def table(rows: list[list[str]]) -> Table:
        built = Table(rows, colWidths=[230, 230], hAlign="LEFT")
        built.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.lightgrey)]))
        return built

    answers = [[label(k), "Not given" if v is None else str(v)] for k, v in profile.model_dump().items()]
    story: list[Flowable] = [
        Paragraph("Life expectancy estimate", styles["Title"]),
        Paragraph(f"Generated {date.today().isoformat()}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(
            f"Middle estimate: age {estimate.median_age:g}. Half of similar people fall between "
            f"{estimate.lower_age:g} and {estimate.upper_age:g}. An average Indian of your age and sex: "
            f"{result.average.median_age:g}.", styles["Heading3"]),
        Paragraph("What moves your estimate (years)", styles["Heading2"]),
        table([[label(f.key), f"{f.years:+g}"] for f in result.factors] or [["No differences", ""]]),
        Paragraph("Suggestions", styles["Heading2"]),
        *[Paragraph(f"<b>{a.title}.</b> {a.detail}", styles["Normal"]) for a in result.advice],
        Paragraph("Your answers", styles["Heading2"]),
        table(answers),
        Spacer(1, 12),
        Paragraph(DISCLAIMER, styles["Italic"]),
    ]
    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="Life expectancy estimate").build(story)
    return buffer.getvalue()
