import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Kalman filter

    A discussion on how Kalman filters work and a short example of its implementation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction

    Motivation: real-time estimation of a fly location to accurate control and image its brain.

    Paper reference: https://www.sciencedirect.com/science/article/pii/S2211124718315766

    Here, the larvae's location was continuous estimated (and updated) using a Kalman filter. This location was used to image the larvae's brain (image below from paper). My aim is to apply a similar approach to estimate the fly location in real-time.

    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2211124718315766-fx1.jpg" width="800" />
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Problem statement

    The aim is to estimate the hidden state of a time varying system with the following properties:

    $$
    x_{t} = F_t x_{t-1} + B_t u_t + w_t
    $$

    where,
    - \(x_t\) is the hidden state at time \(t\)
    - \(F_t\) is the state transition model which is applied to the previous state \(x_{t-1}\)
    - \(B_t\) is the control-input model which is applied to the control vector \(u_t\)
    - \(w_t\) is the **process noise** which is assumed to be drawn from gaussian distribution
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, we do not have access to the hidden state directly. Instead, we have access to a noisy measurement \(z_t\) of the hidden state:

    $$
    z_t = H_t x_t + v_t
    $$

    where,
    - \(z_t\) is the measurement at time \(t\)
    - \(H_t\) is the observation model which maps the true state space into the observed space
    - \(v_t\) is the **measurement noise** which is also assumed to be drawn from gaussian distribution
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Kalman filter overview

    The Kalman filter can predict the hidden states for a system described using the two equations:

    $$
    x_{t} = F_t x_{t-1} + B_t u_t + w_t
    $$

    $$
    z_t = H_t x_t + v_t
    $$


    Based on these, the Kalman filter predicts the hidden state \(x_t\) and its covariance \(P_t\).

    /// admonition | The catch is that, the Kalman filter assumes the following is known:
    - The initial state $x_0$ (reasonable)
    - The control input $u_t$ is known (reasonable)
    - Transition matrixes $F_t$, $B_t$, $H_t$ is known (reasonable?)
    - The process error variance of w_t is known (might be hard to estimate?)
    - The measurement error variance v_t is known (might be hard to estimate?)
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    It seems like a lot needs to known about the system before using a Kalman filter. So, why use it then?

    **The advantage is that Kalman filter is a recursive approximator / filter / predictor.** It runs in constant time and needs only the previous state to estimate the current state. This makes it suitable for real-time applications where data arrives sequentially.

    It runs in two steps:
    1. **Prediction step**: Based on the previous state estimate and the system model, predict the current state and its uncertainty.
    2. **Update step**: Incorporate the new measurement to refine the state estimate and reduce uncertainty.

    Because of this, it is often used in robotics and control applications. Additionally, multiple different types of measurements can be combined to improve the state estimate, making it useful in cases with different types of sensors.

    /// attention | Assumptions:
    Because Kalman filters rely on squared errors to estimate hidden states, they assume that the errors are normally distributed (similar to all least-square error minimization approaches). Therefore, Kalman filters work best when the noise in the system is Gaussian (it also works in some cases even if the noise is not gaussian).
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Train position estimation example

    <img src="public/kalman_filter_01.png"/>

    The hidden state of the train, its position and velocity, needs to be known by the train operator.

    The available measurements are:
    1. velocity of the train (measured using a speedometer)
    2. position of the train (measured through a radio antenna on the roof)
    3. acceleration/deceleration of the train (inputs $u_t$)

    Based on the two noisy measurements, the position and velocity of the train needs to be estimated.

    Image from: https://courses.physics.illinois.edu/ece420/sp2019/7_UnderstandingKalmanFilter.pdf
    """)
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ### Train position estimation steps

    <img src="public/kalman_filter_02.png"/>

    <img src="public/kalman_filter_03.png"/>

    <img src="public/kalman_filter_04.png"/>

    <img src="public/kalman_filter_05.png"/>

    Images from: https://courses.physics.illinois.edu/ece420/sp2019/7_UnderstandingKalmanFilter.pdf
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## State estimation steps

    ![alt](public/kalman_filter_06.png)

    Image from: https://arxiv.org/pdf/1710.04055 (Fig. 4)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Pictorial representation

    ![alt](public/kalman_filter_07.png)

    Image from: https://arxiv.org/pdf/1710.04055 (Fig. 5)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implementation the train example

    Hidden states:

    $$
    x_t = \begin{bmatrix}
    p_t\\
    v_t
    \end{bmatrix}
    $$

    State transition model

    $$
    x_t = \begin{bmatrix}
    1 & \Delta t\\
    0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    p_{t-1}\\
    v_{t-1}
    \end{bmatrix} +
    \begin{bmatrix}
    \frac{\Delta t^2}{2}\\
    \Delta t
    \end{bmatrix} a_t + w_t
    $$

    $$
    x_t = F_t x_{t-1} + B_t u_t + w_t
    $$

    Observation model:

    $$
    z_t = \begin{bmatrix}
    1 & 0\\
    0 & 1
    \end{bmatrix}\begin{bmatrix}
    p_{t}\\
    v_{t}
    \end{bmatrix} + v_t
    $$

    $$
    z_t = H_t x_t + v_t
    $$
    """)
    return


@app.cell
def _(mo, np):
    process_cov_pos = mo.ui.slider(
        steps=np.arange(1000, 100000, 1000),
        value=5000.0,
        label="Process noise variance (position):",
        show_value=True,
    )
    process_cov_vel = mo.ui.slider(
        steps=np.arange(0, 10.1, 0.1),
        value=0.5,
        label="Process noise variance (velocity):",
        show_value=True,
    )
    measurement_cov_pos = mo.ui.slider(
        steps=np.arange(10000, 100000, 10000),
        value=50000.0,
        label="Measurement noise variance (position):",
        show_value=True,
    )
    measurement_cov_vel = mo.ui.slider(
        steps=np.arange(0, 20.1, 0.1),
        value=2.0,
        label="Measurement noise variance (velocity):",
        show_value=True,
    )
    process_cov_pos, process_cov_vel, measurement_cov_pos, measurement_cov_vel
    return (
        measurement_cov_pos,
        measurement_cov_vel,
        process_cov_pos,
        process_cov_vel,
    )


@app.cell
def _(
    measurement_cov_pos,
    measurement_cov_vel,
    np,
    process_cov_pos,
    process_cov_vel,
):
    # Generate data
    rng = np.random.default_rng(42)
    num_steps = 100
    dt = 1.0  # time step

    # True initial state
    x0 = np.array(
        [
            [0],  # initial position
            [0],
        ]
    )  # initial velocity
    P0 = np.array([[0, 0], [0, 0]])  # initial covariance

    # control input (acceleration)
    acceleration = np.concat(
        [
            np.zeros(10),
            2 * np.ones(20),
            np.ones(20),
            np.zeros(10),
            -np.ones(20),
            -2 * np.ones(20),
        ]
    )
    # print(acceleration)

    # State transition model
    F = np.array([[1, dt], [0, 1]])
    # Control-input model
    B = np.array([[0.5 * dt**2], [dt]])

    # True states
    true_states = []
    x = x0
    for a in acceleration:
        x = F @ x + B * a
        true_states.append(x.flatten())
    true_states = np.array(true_states)

    # # Obtain noise variances from sliders
    # process_cov_pos = 1.0  # process noise variance for position
    # process_cov_vel = 0.5  # process noise variance for velocity
    # measurement_cov_pos = 5.0  # measurement noise variance for position
    # measurement_cov_vel = 2.0  # measurement noise variance for velocity

    # Obtain noise
    process_noise = np.array(
        [
            rng.normal(0, np.sqrt(process_cov_pos.value), size=num_steps),
            rng.normal(0, np.sqrt(process_cov_vel.value), size=num_steps),
        ]
    ).transpose()

    measurement_noise = np.array(
        [
            rng.normal(0, np.sqrt(measurement_cov_pos.value), size=num_steps),
            rng.normal(0, np.sqrt(measurement_cov_vel.value), size=num_steps),
        ]
    ).transpose()

    # Noisy states
    noisy_states = true_states + np.cumsum(process_noise, axis=0)

    # Measurements
    measurements = noisy_states + measurement_noise
    return (
        B,
        F,
        P0,
        acceleration,
        measurements,
        noisy_states,
        num_steps,
        true_states,
        x0,
    )


@app.cell
def _(acceleration, measurements, noisy_states, plt, true_states):
    # Plot data
    plt.figure(figsize=(14, 10))

    # Plot Position
    plt.subplot(3, 1, 1)
    plt.plot(true_states[:, 0], label="Position (noiseless model)", color="g")
    plt.plot(
        noisy_states[:, 0],
        label="Position (ground truth)",
        color="r",
        linestyle="--",
    )
    plt.plot(
        measurements[:, 0],
        label="Measurements (process + observational noise)",
        color="b",
        linestyle=":",
    )
    plt.title("Position Estimation")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.legend()

    # Plot Velocity
    plt.subplot(3, 1, 2)
    plt.plot(true_states[:, 1], label="Velocity (noiseless model)", color="g")
    plt.plot(
        noisy_states[:, 1],
        label="Velocity (ground truth)",
        color="r",
        linestyle="--",
    )
    plt.plot(
        measurements[:, 1],
        label="Measurements (process + observational noise)",
        color="b",
        linestyle=":",
    )
    plt.title("Velocity Estimation")
    plt.xlabel("Time Steps")
    plt.ylabel("Velocity")
    plt.legend()

    # Plot Acceleration
    plt.subplot(3, 1, 3)
    plt.plot(acceleration, label="acceleration", color="b", linestyle=":")
    plt.title("Acceleration")
    plt.xlabel("Time Steps")
    plt.ylabel("Acceleration")
    plt.legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(np):
    class KalmanFilter:
        def __init__(self, F, B, H, Q, R, x0, P0):
            self.F = F  # State transition model
            self.B = B  # Control-input model
            self.H = H  # Observation model
            self.Q = Q  # Process noise covariance
            self.R = R  # Measurement noise covariance
            self.x = x0  # Initial state estimate
            self.x_pred = None  # Predicted state estimate
            self.P = P0  # Initial covariance estimate
            self.P_pred = None  # Predicted covariance estimate

        def predict(self, u):
            # Predict the next state
            self.x_pred = self.F @ self.x + self.B @ u
            self.P_pred = self.F @ self.P @ self.F.T + self.Q

        def update(self, z):
            # Update the state with measurement z
            y = z - self.H @ self.x_pred  # Measurement residual
            S = self.H @ self.P_pred @ self.H.T + self.R  # Residual covariance
            K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            self.x = self.x + K @ y
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ self.H) @ self.P_pred
    return (KalmanFilter,)


@app.cell
def _(
    measurement_cov_pos,
    measurement_cov_vel,
    mo,
    process_cov_pos,
    process_cov_vel,
):
    use_data_cov = mo.ui.checkbox(
        label="Use data noise variances as guesses for Kalman filter?",
        value=False,
    )

    g_process_cov_pos = mo.ui.slider(
        steps=process_cov_pos.steps,
        value=2000.0,
        label="Guess of process noise variance (position):",
        show_value=True,
    )
    g_process_cov_vel = mo.ui.slider(
        steps=process_cov_vel.steps,
        value=1.0,
        label="Guess of process noise variance (velocity):",
        show_value=True,
    )
    g_measurement_cov_pos = mo.ui.slider(
        steps=measurement_cov_pos.steps,
        value=70000.0,
        label="Guess of measurement noise variance (position):",
        show_value=True,
    )
    g_measurement_cov_vel = mo.ui.slider(
        steps=measurement_cov_vel.steps,
        value=4.0,
        label="Guess of measurement noise variance (velocity):",
        show_value=True,
    )

    (
        use_data_cov,
        g_process_cov_pos,
        g_process_cov_vel,
        g_measurement_cov_pos,
        g_measurement_cov_vel,
    )
    return (
        g_measurement_cov_pos,
        g_measurement_cov_vel,
        g_process_cov_pos,
        g_process_cov_vel,
        use_data_cov,
    )


@app.cell
def _(
    B,
    F,
    KalmanFilter,
    P0,
    g_measurement_cov_pos,
    g_measurement_cov_vel,
    g_process_cov_pos,
    g_process_cov_vel,
    measurement_cov_pos,
    measurement_cov_vel,
    np,
    process_cov_pos,
    process_cov_vel,
    use_data_cov,
    x0,
):
    if use_data_cov.value:
        train_kalman_filter = KalmanFilter(
            F=F,
            B=B,
            # H=np.array([[1, 0], [0, 0]]),
            H=np.eye(2),
            Q=np.array([[process_cov_pos.value, 0], [0, process_cov_vel.value]]),
            R=np.array(
                [
                    [measurement_cov_pos.value, 0],
                    [0, measurement_cov_vel.value],
                ]
            ),
            x0=x0,
            P0=P0,
        )

    else:
        train_kalman_filter = KalmanFilter(
            F=F,
            B=B,
            # H=np.array([[1, 0], [0, 0]]),
            H=np.eye(2),
            Q=np.array(
                [[g_process_cov_pos.value, 0], [0, g_process_cov_vel.value]]
            ),
            R=np.array(
                [
                    [g_measurement_cov_pos.value, 0],
                    [0, g_measurement_cov_vel.value],
                ]
            ),
            x0=x0,
            P0=P0,
        )
    return (train_kalman_filter,)


@app.cell
def _(acceleration, measurements, np, num_steps, train_kalman_filter):
    estimated_states = []
    for j in range(num_steps):
        u = np.array([[acceleration[j]]])  # control input (acceleration)
        z = measurements[j].reshape(-1, 1)  # measurement

        train_kalman_filter.predict(u)
        train_kalman_filter.update(z)

        estimated_states.append(train_kalman_filter.x.flatten())
    estimated_states = np.array(estimated_states)
    return (estimated_states,)


@app.cell
def _(estimated_states, measurements, noisy_states, plt, true_states):
    # Plot estimated states

    plt.figure(figsize=(14, 10))
    # Plot Position
    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], label="Position (noiseless model)", color="g")
    plt.plot(
        noisy_states[:, 0],
        label="Position (ground truth)",
        color="r",
        linestyle="--",
    )
    plt.plot(
        measurements[:, 0],
        label="Measurements (process + observational noise)",
        color="b",
        linestyle=":",
    )
    plt.plot(
        estimated_states[:, 0],
        label="Estimated Position (Kalman filter)",
        color="orange",
        linestyle="-.",
    )
    plt.title("Position Estimation with Kalman Filter")
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.legend()

    # Plot Velocity
    plt.subplot(2, 1, 2)
    plt.plot(true_states[:, 1], label="Velocity (noiseless model)", color="g")
    plt.plot(
        noisy_states[:, 1],
        label="Velocity (ground truth)",
        color="r",
        linestyle="--",
    )
    plt.plot(
        measurements[:, 1],
        label="Measurements (process + observational noise)",
        color="b",
        linestyle=":",
    )
    plt.plot(
        estimated_states[:, 1],
        label="Estimated Velocity (Kalman filter)",
        color="orange",
        linestyle="-.",
    )
    plt.title("Velocity Estimation with Kalman Filter")
    plt.xlabel("Time Steps")
    plt.ylabel("Velocity")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
