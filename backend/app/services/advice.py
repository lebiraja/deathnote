"""Rule-based suggestions, kept separate from the model's numbers."""

from app.schemas.prediction import Advice, Profile


def advise(profile: Profile, sleep_hours: float, bmi: float) -> list[Advice]:
    rules: list[tuple[bool, str, str]] = [
        (profile.smoking == "current", "Stop smoking",
         "This is usually the largest change available. A doctor can help with nicotine replacement or medicine."),
        (profile.chews_tobacco, "Stop chewing tobacco",
         "Gutka, khaini and paan with tobacco raise the risk of death and of mouth cancer."),
        (sleep_hours < 7, "Make room for more sleep",
         "Adults are generally advised to get at least 7 hours. Try an earlier, regular bedtime."),
        (sleep_hours > 9, "Check on long sleep",
         "Regularly sleeping over 9 hours can go with health problems. Mention it to a doctor."),
        (profile.moderate_minutes + 2 * profile.vigorous_minutes < 150, "Move more",
         "Aim for 150 minutes of brisk walking or similar each week."),
        (profile.alcohol == "heavy", "Drink less",
         "Cutting down lowers the risk of liver disease, injury and some cancers."),
        (profile.diet in ("fair", "poor"), "Improve what you eat",
         "More vegetables, fruit, pulses and whole grains; less fried and packaged food and less salt."),
        (bmi >= 23, "Watch your weight",
         "For Indian adults, a BMI of 23 or more counts as overweight. Check your waist and blood sugar too."),
        (bmi < 18.5, "Check low weight", "A BMI under 18.5 is worth discussing with a doctor."),
        (profile.systolic_bp is None or profile.total_cholesterol is None, "Know your numbers",
         "A routine check of blood pressure, blood sugar and cholesterol shows risks you cannot feel."),
        (profile.diabetes or profile.hypertension or profile.heart_disease, "Keep treatment going",
         "Take prescribed medicines and keep follow-up visits."),
    ]
    return [Advice(title=title, detail=detail) for applies, title, detail in rules if applies]
