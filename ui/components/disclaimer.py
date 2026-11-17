"""Persistent research-use notice."""


DISCLAIMER = (
    "Research use only: outputs are exploratory and are not clinical diagnoses. "
    "Displayed probabilities are model outputs, not calibrated confidence. "
    "Participant identities are not verified, so subject-independent "
    "generalization has not been established."
)


def render_research_disclaimer(container):
    container.warning(DISCLAIMER)
