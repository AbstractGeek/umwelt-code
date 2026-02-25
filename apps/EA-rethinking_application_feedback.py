"""Rethinking Application Feedback

A marimo app that simulates a multi-stage hiring process and provides different
approaches to communicating feedback to applicants. Visualizes how applicants
rank against the distribution of scores at each stage.
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pandas as pd

    return mo, np, pd, plt, stats


@app.cell
def _(mo):
    """Configure simulation parameters and applicant selection."""

    # Simulation controls
    applications = mo.ui.number(
        start=10, stop=10000, step=10, value=1000, label="Applications received: "
    )
    first_stage = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=10,
        label="First stage (%): ",
    )  # percentage of applications that pass the first stage
    second_stage = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=5,
        label="Second stage (%): ",
    )  # percentage of applications that pass the second stage
    hiring_positions = mo.ui.number(
        start=1, stop=10, step=1, value=2, label="Number of positions to hire: "
    )  # number of positions to hire (i.e. top N candidates from final stage)

    # random_seed = mo.ui.number(start=10, stop=10000, step=10, value=1000)
    return applications, first_stage, hiring_positions, second_stage


@app.cell
def _(applications, first_stage, hiring_positions, np, pd, second_stage):
    """Generate simulated applicant data across three interview stages.

    Each stage uses different statistical distributions to represent realistic
    patterns and to demonstrate how distribution-based feedback helps candidates
    understand their performance relative to the applicant pool.
    """

    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed=2 * 42)

    # STAGE 1: Application + Resume Review
    # Uses lognormal distribution (long right tail) to model many applicants with fewer strong candidates
    names = [f"Applicant {i + 1}" for i in range(applications.value)]
    application_scores = np.round(
        np.clip(
            rng.lognormal(mean=3.0, sigma=0.5, size=applications.value), 1, 100
        )
        / 10
    )
    resume_scores = np.round(
        np.clip(
            rng.lognormal(mean=3.0, sigma=0.5, size=applications.value), 1, 100
        )
        / 10
    )
    stage_1_scores = np.round((2 * application_scores + resume_scores) / 3, 1)
    stage_1_cutoff = np.percentile(stage_1_scores, 100 - first_stage.value)
    passed_stage_1 = stage_1_scores >= stage_1_cutoff

    # STAGE 2: Work Test + First Interview
    # Work test: t-distribution (like normal but with heavier tails)
    # Interview: inverted lognormal (subtracting from 10 creates left-skew distribution)
    worktest_scores = np.round(
        np.clip(
            5 + 5 * (rng.standard_t(df=2, size=applications.value) / 10), 0, 10
        )
    )
    worktest_scores[~passed_stage_1] = 0
    first_interview_scores = np.round(
        np.clip(
            10
            - np.round(rng.lognormal(mean=3, sigma=0.5, size=applications.value))
            / 10,
            0,
            100,
        )
    )
    first_interview_scores[~passed_stage_1] = 0
    stage_2_scores = np.round((worktest_scores + first_interview_scores) / 2, 1)
    stage_2_cutoff = np.percentile(
        stage_2_scores[passed_stage_1], 100 - second_stage.value
    )
    passed_stage_2 = stage_2_scores >= stage_2_cutoff


    # STAGE 3: Final Interview
    # Normal distribution (candidates who reach here are generally strong)
    final_interview_scores = (
        np.clip(
            np.round(rng.normal(loc=10, scale=3, size=applications.value)), 0, 20
        )
        / 2
    )
    final_interview_scores[~passed_stage_2] = 0

    # Select top candidates for hiring positions
    selected = np.full_like(final_interview_scores, False, dtype=bool)
    selected[final_interview_scores.argsort()[::-1][: hiring_positions.value]] = (
        True
    )

    # Calculate total score (sum of all stages)
    total_scores = stage_1_scores + stage_2_scores + final_interview_scores

    # Create DataFrame with all candidate data
    applications_df = pd.DataFrame(
        {
            "Name": names,
            "Application Score": application_scores,
            "Resume Score": resume_scores,
            "First Stage Score": stage_1_scores,
            "Selected First Stage": passed_stage_1,
            "Work Test Score": worktest_scores,
            "First Interview Score": first_interview_scores,
            "Second Stage Score": stage_2_scores,
            "Selected Second Stage": passed_stage_2,
            "Final Interview Score": final_interview_scores,
            "Total Score": total_scores,
            "Normalized Score": np.round(total_scores / np.max(total_scores), 4),
            "Hired": selected,
        }
    )
    return applications_df, names


@app.cell
def _(applications, applications_df, mo, names, np, pd):
    """Create sorted view and normalize scores for visualization.

    Generates rankings and percentiles for each scoring stage to enable
    comparative feedback and ranking displays.
    """

    # Create sorted view for database display
    sorted_df = applications_df.sort_values(
        [
            "Total Score",
            "Final Interview Score",
            "Second Stage Score",
            "First Stage Score",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # Create normalized scores, percentiles, and rankings for each stage
    normalized_df = pd.DataFrame(
        {
            "Name": names,
            "First Stage Score": np.round(
                applications_df["First Stage Score"]
                / applications_df["First Stage Score"].max(),
                4,
            ),
            "First Stage Percentile": applications_df["First Stage Score"]
            .rank(pct=True)
            .round(4),
            "First Stage Position": applications_df["First Stage Score"]
            .rank(ascending=False)
            .astype(int),
            "Selected First Stage": applications_df["Selected First Stage"],
            "Second Stage Score": np.round(
                applications_df["Second Stage Score"]
                / applications_df["Second Stage Score"].max(),
                4,
            ),
            "Second Stage Percentile": applications_df["Second Stage Score"]
            .rank(pct=True)
            .round(4),
            "Second Stage Position": applications_df["Second Stage Score"]
            .rank(ascending=False)
            .astype(int),
            "Selected Second Stage": applications_df["Selected Second Stage"],
            "Final Interview Score": np.round(
                applications_df["Final Interview Score"]
                / applications_df["Final Interview Score"].max(),
                4,
            ),
            "Final Interview Percentile": applications_df["Final Interview Score"]
            .rank(pct=True)
            .round(4),
            "Final Interview Position": applications_df["Final Interview Score"]
            .rank(ascending=False)
            .astype(int),
            "Hired": applications_df["Hired"],
            "Total Score": np.round(
                applications_df["Total Score"]
                / applications_df["Total Score"].max(),
                4,
            ),
            "Total Percentile": applications_df["Total Score"]
            .rank(pct=True)
            .round(4),
            "Total Position": applications_df["Total Score"]
            .rank(ascending=False)
            .astype(int),
        }
    )

    ## Select applicant
    selected_applicant = mo.ui.slider(
        start=1,
        stop=applications.value,
        step=1,
        value=50,
        include_input=True,
        label="Select an application number: ",
    )
    return normalized_df, selected_applicant, sorted_df


@app.cell
def _(applications_df, normalized_df, selected_applicant):
    """Extract data for the selected applicant from raw and normalized views."""

    sel_app = applications_df.loc[selected_applicant.value - 1]
    n_app = normalized_df.loc[selected_applicant.value - 1]
    return n_app, sel_app


@app.cell
def _(applications, hiring_positions, mo, n_app):
    """Display Approach 1: Simple ranking and percentile statistics."""

    approach_1_header = mo.md(rf"""
    ## Approach 1: Simple Statistics
    """)

    approach_1_text = mo.vstack(
        [
            mo.md(rf"Total applications: {applications.value}"),
            mo.md(rf"Positions available: {hiring_positions.value}"),
            mo.md("-"),
            mo.md(
                rf"Applicant's rank: **{n_app['Total Position'] + 1}** (top {hiring_positions.value} are selected)"
            ),
            mo.md(
                rf"Applicant's percentile: **{n_app['Total Percentile']:.2%}**th percentile"
            ),
        ],
        gap=0,
    )
    return approach_1_header, approach_1_text


@app.cell
def _(applications_df, mo, np, plt, sel_app, stats):
    """Display Approach 2: Distribution visualization with histogram and KDE.

    Shows the distribution of total scores across all applicants, with markers
    for the selected applicant and hired candidates.
    """

    # Color scheme for visualizations
    hist_color = "#8ecae6"  # Light blue for histogram
    hist_edge = "#023047"  # Dark blue for edges
    kde_color = "#1f77b4"  # Blue for KDE curve
    marker_color = "#f93200"  # Red for selected applicant
    selected_color = "#06d6a0"  # Green for hired candidates

    # Create figure with histogram and KDE
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _scores = applications_df["Total Score"].values
    # Plot histogram of score distribution
    _ax.hist(
        _scores,
        bins=30,
        color=hist_color,
        edgecolor=hist_edge,
        linewidth=0.8,
        alpha=0.45,
        density=False,
    )

    # Add KDE curve on secondary y-axis for density
    _kde = stats.gaussian_kde(_scores)
    _x_grid = np.linspace(_scores.min(), _scores.max(), 200)
    _ax2 = _ax.twinx()
    _ax2.plot(_x_grid, _kde(_x_grid), color=kde_color, linewidth=2.2)
    _ax2.fill_between(_x_grid, _kde(_x_grid), color=kde_color, alpha=0.15)

    # Mark selected applicant
    _ax.axvline(
        sel_app["Total Score"],
        color=marker_color,
        linestyle="--",
        linewidth=2,
        label="Applicant",
    )

    # Mark hired candidates with vertical line segments
    for idx, _score in enumerate(
        applications_df.loc[applications_df["Hired"]]["Total Score"]
    ):
        _ax.plot(
            [_score, _score],
            [0, _ax.get_ylim()[1] * 0.25],
            color=selected_color,
            linestyle="--",
            linewidth=2,
            label=f"Hired {idx + 1}",
        )

    _ax.set_xlabel("Total Score")
    _ax.set_ylabel("Frequency")
    _ax2.set_ylabel("Density")
    _ax.set_ylim(bottom=0)
    _ax2.set_ylim(bottom=0)
    _ax.set_title("Distribution of Total Scores")
    _ax.grid(axis="y", alpha=0.25)
    _ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()

    # Create markdown header and figure embed
    approach_2_header = mo.md(rf"""
    ## Approach 2: Visual Statistics
    """)

    approach_2_text = mo.md(rf"""
    {mo.as_html(_fig)}
    """)
    return (
        approach_2_header,
        approach_2_text,
        hist_color,
        hist_edge,
        kde_color,
        marker_color,
        selected_color,
    )


@app.cell
def _(mo, n_app, sel_app):
    """Display Approach 3 summary text (distributions shown below)."""

    approach_3_header = mo.md(rf"""
    ## Approach 3: Descriptive Statistics
    """)

    approach_3_text = [
        mo.vstack(
            [
                # mo.md(rf"Total applications: {applications.value}"),
                # mo.md(rf"Positions available: {hiring_positions.value}"),
                # mo.md("-"),
                mo.md("### Stage 1: Application + Resume Scores"),
                mo.md(
                    "Applicant status: "
                    + (
                        "Selected"
                        if sel_app["Selected First Stage"]
                        else "Not selected"
                    )
                ),
                mo.md(rf"Applicant's rank: {n_app['First Stage Position']}"),
                mo.md(
                    rf"Applicant's percentile: {n_app['First Stage Percentile']:.2%}"
                ),
            ],
            gap=0,
        ),
        mo.vstack(
            [
                mo.md("### Stage 2: First interview + Work Test Scores"),
                mo.md(
                    "Applicant status: "
                    + (
                        "Selected"
                        if sel_app["Selected Second Stage"]
                        else "Not selected"
                        if sel_app["Selected First Stage"]
                        else "N/A"
                    )
                ),
                mo.md(
                    "Applicant's rank: "
                    + (
                        f"{n_app['Second Stage Position']}"
                        if sel_app["Selected First Stage"]
                        else "N/A"
                    )
                ),
                mo.md(
                    "Applicant's percentile: "
                    + (
                        f"{n_app['Second Stage Percentile']:.2%}"
                        if sel_app["Selected First Stage"]
                        else "N/A"
                    )
                ),
            ],
            gap=0,
        ),
        mo.vstack(
            [
                mo.md("### Stage 3: Final Interview Scores"),
                mo.md(
                    "Applicant status: "
                    + (
                        "Hired"
                        if sel_app["Hired"]
                        else "Not hired"
                        if sel_app["Selected Second Stage"]
                        else "N/A"
                    )
                ),
                mo.md(
                    "Applicant's rank: "
                    + (
                        f"{n_app['Final Interview Position']}"
                        if sel_app["Selected Second Stage"]
                        else "N/A"
                    )
                ),
                mo.md(
                    "Applicant's percentile: "
                    + (
                        f"{n_app['Final Interview Percentile']:.2%}"
                        if sel_app["Selected Second Stage"]
                        else "N/A"
                    )
                ),
            ],
            gap=0,
        ),
    ]
    return approach_3_header, approach_3_text


@app.cell
def _(
    hist_color,
    hist_edge,
    kde_color,
    marker_color,
    mo,
    n_app,
    normalized_df,
    np,
    plt,
    selected_color,
    stats,
):
    """Display Approach 3: Three-panel visualization of scores at each stage.

    Shows normalized score distributions for each interview stage, comparing
    selected vs. non-selected candidates, with KDE curves and selected applicant marker.
    """

    # approach 3 figures
    approach_3_figures = []

    # Create three separate figures, one for each stage
    for _idx, (col, pass_col, title) in enumerate(
        [
            ("First Stage Score", "Selected First Stage", "Application + Resume"),
            (
                "Second Stage Score",
                "Selected Second Stage",
                "First interview + Work Test",
            ),
            ("Final Interview Score", "Hired", "Final Interview"),
        ]
    ):
        _fig, _ax = plt.subplots(figsize=(8, 4))

        # Get valid scores (> 0) for current stage
        _scores = normalized_df[col][normalized_df[col] > 0].values
        _, _bin_edges = np.histogram(_scores, bins=30)

        # Separate selected and non-selected candidates
        _scores_unselected = normalized_df[col][
            (normalized_df[col] > 0) & (~normalized_df[pass_col])
        ].values
        _scores_selected = normalized_df[col][normalized_df[pass_col]].values
        _counts_unselected, _ = np.histogram(_scores_unselected, bins=_bin_edges)
        _counts_selected, _ = np.histogram(_scores_selected, bins=_bin_edges)

        # Plot unselected candidates
        _ax.stairs(
            _counts_unselected,
            _bin_edges,
            fill=True,
            color=hist_color,
            edgecolor=hist_edge,
            linewidth=0.8,
            alpha=0.45,
        )

        # Plot selected/hired candidates
        _ax.stairs(
            _counts_selected,
            _bin_edges,
            fill=True,
            color=selected_color,
            edgecolor=hist_edge,
            linewidth=0.8,
            alpha=0.45,
            label="Hired" if title == "Final Interview" else "Selected",
        )

        # Overlay KDE curve scaled to match histogram frequency
        _kde = stats.gaussian_kde(_scores)
        _x_grid = np.linspace(_scores.min(), _scores.max(), 200)
        _kde_scaled = _kde(_x_grid) * len(_scores) * np.diff(_bin_edges[:2])
        _ax.plot(_x_grid, _kde_scaled, color=kde_color, linewidth=2.2)
        _ax.fill_between(_x_grid, _kde_scaled, color=kde_color, alpha=0.15)

        _ax.set_xlabel("Score")
        _ax.set_ylabel("Frequency")
        _ax.set_title(title)
        _ax.grid(axis="y", alpha=0.25)

        # Add vertical line for selected applicant's score
        _applicant_score = n_app[col]
        if _applicant_score != 0:
            _ax.axvline(
                _applicant_score,
                color=marker_color,
                linestyle="--",
                linewidth=2,
                label="Applicant",
            )
        _ax.legend(frameon=False, loc="upper right")
        plt.tight_layout()
        plt.show()

        # Store html figure
        # approach_3_figures.append(rf"""
        # {mo.as_html(_fig)}
        # """)
        approach_3_figures.append(mo.as_html(_fig))

    plt.tight_layout()
    plt.show()
    return (approach_3_figures,)


@app.cell
def _(
    applications,
    approach_1_header,
    approach_1_text,
    approach_2_header,
    approach_2_text,
    approach_3_figures,
    approach_3_header,
    approach_3_text,
    first_stage,
    hiring_positions,
    mo,
    second_stage,
    selected_applicant,
    sorted_df,
):
    """Assemble the app UI with three tabs for different feedback approaches."""

    # Parameters tab: simulation controls and candidate database
    tab1 = mo.vstack(
        [
            mo.md(r"""
            ### Simulation parameters

            """),
            mo.vstack(
                [
                    applications,
                    first_stage,
                    second_stage,
                    hiring_positions,
                ]
            ),
            mo.md("### Generated application database"),
            mo.as_html(sorted_df),
            mo.md(
                "*Note:* Selection is based on Final interview score. Total score is calculated for sharing with applicants. "
            ),
        ],
        gap=2,
    )

    # Approach 1 tab: simple ranking statistics
    tab2 = mo.vstack(
        [
            approach_1_header,
            selected_applicant,
            approach_1_text,
        ],
        gap=2,
    )

    # Approach 2 tab: visual distribution with applicant marker
    tab3 = mo.vstack(
        [approach_2_header, selected_applicant, approach_1_text, approach_2_text],
        gap=2,
    )

    # Approach 3 tab: descriptive statistics and three-panel distribution
    tab4 = mo.vstack(
        [
            approach_3_header,
            selected_applicant,
            approach_1_text,
            approach_3_text[0],
            approach_3_figures[0],
            approach_3_text[1],
            approach_3_figures[1],
            approach_3_text[2],
            approach_3_figures[2],
        ],
        gap=2,
    )

    # Create tabbed interface
    tabs = mo.ui.tabs(
        {
            "Parameters": tab1,
            "Approach 1": tab2,
            "Approach 2": tab3,
            "Approach 3": tab4,
        }
    )
    return (tabs,)


@app.cell(hide_code=True)
def _(mo, tabs):
    """Display the main app interface."""

    mo.md(rf"""
    # Rethinking Application Feedback

    {tabs}
    """)
    return


if __name__ == "__main__":
    app.run()
