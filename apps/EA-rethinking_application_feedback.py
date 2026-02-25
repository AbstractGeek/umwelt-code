import marimo

__generated_with = "0.19.4"
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
    # Define constants for the simulation

    ## User inputs
    applications = mo.ui.number(
        start=10, stop=10000, step=10, value=1000, label="Applications received: "
    )  # number of applications to simulate
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

    ## Select applicant
    selected_applicant = mo.ui.number(
        start=1,
        stop=applications.stop,
        step=1,
        value=50,
        label="Select an application number: ",
    )
    return (
        applications,
        first_stage,
        hiring_positions,
        second_stage,
        selected_applicant,
    )


@app.cell
def _(applications, first_stage, hiring_positions, np, pd, second_stage):
    # Generate applications
    # Note: Different stages use different sampling distributions to show the usefulness of distribution based feedback to applicants.

    ## Set rng seed for reproducibility
    rng = np.random.default_rng(seed=2 * 42)

    ## Stage 1: Application and Resume Scores
    # Application and resume score are lognormally distributed (long tail to the right; mimicking lot of applicants but fewer good ones)
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

    ## Stage 2: Work Test and Interview Scores
    # Work test scores is t-distributed (gaussian-like, but with heavy tails)
    # Interview scores are lognormally distributed but with long tail on the left (i.e. all candidates are generally good at interviewing)
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


    ## Stage 3: Final Interview
    # Final interview scores are normally distributed (i.e. all are generally good)
    final_interview_scores = (
        np.clip(
            np.round(rng.normal(loc=10, scale=3, size=applications.value)), 0, 20
        )
        / 2
    )
    final_interview_scores[~passed_stage_2] = 0
    sorted_final_scores_ind = final_interview_scores.argsort()
    selected_ind = sorted_final_scores_ind[
        ~np.isnan(final_interview_scores[sorted_final_scores_ind])
    ][::-1][: hiring_positions.value]
    selected = np.full_like(final_interview_scores, False, dtype=bool)
    selected[final_interview_scores.argsort()[::-1][: hiring_positions.value]] = (
        True
    )

    total_scores = stage_1_scores + stage_2_scores + final_interview_scores

    applications_df = pd.DataFrame(
        {
            "Name": names,
            "Application Score": application_scores,
            "Resume Score": resume_scores,
            "First Stage Score": stage_1_scores,
            "Passed First Stage": passed_stage_1,
            "Work Test Score": worktest_scores,
            "First Interview Score": first_interview_scores,
            "Second Stage Score": stage_2_scores,
            "Passed Second Stage": passed_stage_2,
            "Final Interview Score": final_interview_scores,
            "Total Score": total_scores,
            "Normalized Score": np.round(total_scores / np.max(total_scores), 4),
            "Selected": selected,
        }
    )
    # applications_df
    return applications_df, names


@app.cell
def _(applications_df, names, np, pd):
    # Generate sorted and normalized scores

    sorted_df = applications_df.sort_values(
        [
            "Total Score",
            "Final Interview Score",
            "Second Stage Score",
            "First Stage Score",
        ],
        ascending=False,
    ).reset_index(drop=True)

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
            "Passed First Stage": applications_df["Passed First Stage"],
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
            "Passed Second Stage": applications_df["Passed Second Stage"],
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
            "Selected": applications_df["Selected"],
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
    # normalized_df
    return normalized_df, sorted_df


@app.cell
def _(normalized_df):
    normalized_df.sort_values("Total Position").head(20)
    return


@app.cell
def _(applications_df, normalized_df, selected_applicant):
    sel_app = applications_df.loc[selected_applicant.value - 1]
    n_app = normalized_df.loc[selected_applicant.value - 1]
    return n_app, sel_app


@app.cell
def _():
    # # Calculate selected applicant's percentile and position for all approaches
    # curr_app = applications_df.loc[selected_applicant.value - 1]

    # ## Approach 1, 2
    # total_percentile = np.round(
    #     stats.percentileofscore(
    #         applications_df["Total Score"], curr_app["Total Score"]
    #     ),
    #     1,
    # )

    # total_position = sorted_df.index[
    #     sorted_df["Name"] == f"Applicant {selected_applicant.value}"
    # ].values[0]

    # ## Approach 3
    # first_stage_percentile = np.round(
    #     stats.percentileofscore(
    #         applications_df["First Stage Score"], curr_app["First Stage Score"]
    #     ),
    #     1,
    # )
    # second_stage_percentile = np.round(
    #     stats.percentileofscore(
    #         applications_df["Second Stage Score"], curr_app["Second Stage Score"]
    #     ),
    #     1,
    # )
    # final_interview_percentile = np.round(
    #     stats.percentileofscore(
    #         applications_df["Final Interview Score"],
    #         curr_app["Final Interview Score"],
    #     ),
    #     1,
    # )
    return


@app.cell
def _(applications, hiring_positions, mo, n_app):
    # Approach 1: Simple Statistics

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
                rf"Applicant's percentile: **{n_app['Total Percentile']}**th percentile"
            ),
            mo.md("-"),
        ],
        gap=0,
    )
    return approach_1_header, approach_1_text


@app.cell
def _(applications_df, mo, np, plt, sel_app, stats):
    # Approach 2: Visual statistics

    ## Colors
    hist_color = "#8ecae6"
    hist_edge = "#023047"
    kde_color = "#1f77b4"
    marker_color = "#f93200"
    selected_color = "#06d6a0"

    ## Plot
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _x = applications_df["Total Score"].values
    _ax.hist(
        _x,
        bins=30,
        color=hist_color,
        edgecolor=hist_edge,
        linewidth=0.8,
        alpha=0.45,
        density=False,
    )
    _kde = stats.gaussian_kde(_x)
    _x_grid = np.linspace(_x.min(), _x.max(), 200)
    _ax2 = _ax.twinx()
    _ax2.plot(_x_grid, _kde(_x_grid), color=kde_color, linewidth=2.2)
    _ax2.fill_between(_x_grid, _kde(_x_grid), color=kde_color, alpha=0.15)
    _ax.axvline(
        sel_app["Total Score"],
        color=marker_color,
        linestyle="--",
        linewidth=2,
        label="Applicant",
    )
    for _sel_i, _sel_score in enumerate(
        applications_df.loc[applications_df["Selected"]]["Total Score"]
    ):
        _ax.plot(
            [_sel_score, _sel_score],
            [0, _ax.get_ylim()[1] * 0.25],  # Line goes from 0 to 80% of max height
            color=selected_color,
            linestyle="--",
            linewidth=2,
            label=f"Selected {_sel_i + 1}",
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

    ## Markdown
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
def _(applications, hiring_positions, mo, sel_app):
    ## Approach 3: Descriptive Statistics

    approach_3_header = mo.md(rf"""
    ## Approach 3: Descriptive Statistics
    """)

    approach_3_text = mo.vstack(
        [
            mo.md(rf"Total applications: {applications.value}"),
            mo.md(rf"Positions available: {hiring_positions.value}"),
            mo.md("-"),
            mo.md("Stage 1: Application + Resume Scores"),
            mo.md(
                "Applicant status: "
                + ("Selected" if sel_app["Passed First Stage"] else "Not selected")
            ),
            mo.md("Applicant rank: "),
            mo.md("-"),
            mo.md("Stage 2: First interview + Work Test Scores"),
            mo.md("-"),
        ],
        gap=0,
    )
    approach_3_text
    return


@app.cell
def _(
    hist_color,
    hist_edge,
    kde_color,
    marker_color,
    n_app,
    normalized_df,
    np,
    plt,
    selected_color,
    stats,
):
    ## Approach 3: Descriptive Statistics

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    # hist_color = "#8ecae6"
    # hist_edge = "#023047"
    # kde_color = "#1f77b4"

    # Plot histograms and KDEs for each stage score
    for _ax, col, pass_col, title in [
        (
            _axes[0],
            "First Stage Score",
            "Passed First Stage",
            "Application + Resume",
        ),
        (
            _axes[1],
            "Second Stage Score",
            "Passed Second Stage",
            "First interview + Work Test",
        ),
        (_axes[2], "Final Interview Score", "Selected", "Final Interview"),
    ]:
        _x = normalized_df[col][normalized_df[col] > 0].values
        _, _bin_edges = np.histogram(_x, bins=30)

        _x_unsel = normalized_df[col][
            (normalized_df[col] > 0) & (~normalized_df[pass_col])
        ].values
        _x_sel = normalized_df[col][normalized_df[pass_col]].values
        _counts, _ = np.histogram(_x_unsel, bins=_bin_edges)
        _counts_sel, _ = np.histogram(_x_sel, bins=_bin_edges)

        _ax.stairs(
            _counts,
            _bin_edges,
            fill=True,
            color=hist_color,
            edgecolor=hist_edge,
            linewidth=0.8,
            alpha=0.45,
        )
        _ax.stairs(
            _counts_sel,
            _bin_edges,
            fill=True,
            color=selected_color,
            edgecolor=hist_edge,
            linewidth=0.8,
            alpha=0.45,
            label="Hired" if title == "Final Interview" else "Selected",
        )

        _kde = stats.gaussian_kde(_x)
        _x_grid = np.linspace(_x.min(), _x.max(), 200)
        _scaled_kde = (
            _kde(_x_grid) * len(_x) * np.diff(_bin_edges[:2])
        )  # Scale KDE to match histogram
        _ax.plot(_x_grid, _scaled_kde, color=kde_color, linewidth=2.2)
        _ax.fill_between(_x_grid, _scaled_kde, color=kde_color, alpha=0.15)
        _ax.set_xlabel("Score")
        _ax.set_ylabel("Frequency")
        _ax.set_title(title)
        _ax.grid(axis="y", alpha=0.25)

    # Plot another histogram in red for the selected applicant's scores across stages


    # Add vertical line for selected applicant's score in each subplot
    selected_applicant_scores = n_app[
        ["First Stage Score", "Second Stage Score", "Final Interview Score"]
    ].values
    for _ax, score in zip(_axes, selected_applicant_scores):
        if score != 0:    
            _ax.axvline(
                score,
                color=marker_color,
                linestyle="--",
                linewidth=2,
                label="Applicant",
            )
        _ax.legend(frameon=False, loc="upper right")


    plt.tight_layout()
    plt.show()

    # approach_3_text = mo.md(rf"""
    # {mo.as_html(_fig)}
    # """)
    return (selected_applicant_scores,)


@app.cell
def _(selected_applicant_scores):
    selected_applicant_scores
    return


@app.cell
def _(
    applications,
    approach_1_header,
    approach_1_text,
    approach_2_header,
    approach_2_text,
    first_stage,
    hiring_positions,
    mo,
    second_stage,
    selected_applicant,
    sorted_df,
):
    ## Make app

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

    tab2 = mo.vstack(
        [
            approach_1_header,
            selected_applicant,
            approach_1_text,
        ],
        gap=2,
    )

    tab3 = mo.vstack(
        [approach_2_header, selected_applicant, approach_1_text, approach_2_text],
        gap=2,
    )

    tabs = mo.ui.tabs(
        {
            "Parameters": tab1,
            "Approach 1": tab2,
            "Approach 2": tab3,
        }
    )
    return (tabs,)


@app.cell(hide_code=True)
def _(mo, tabs):
    mo.md(rf"""
    # Rethinking Application Feedback

    {tabs}
    """)
    return


if __name__ == "__main__":
    app.run()
