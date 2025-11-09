"""
Physically corrected robot simulator with proper dimensional analysis.
Fixes critical issues identified in paper review.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import TEST_PID_PNG

class RobotSimulator:
    """
    Physically accurate robot model with correct dimensional analysis.

    Physical parameters:
    - mass: [kg] - translational mass
    - damping_coeff: [N·s/m] - viscous damping coefficient (not friction!)
    - inertia: [kg·m²] - rotational moment of inertia
    - radius: [m] - characteristic radius for rotation-translation coupling
    """

    def __init__(self, mass=1.0, damping_coeff=0.5, inertia=0.1, radius=0.1):
        """
        Initialize robot with physical parameters.

        Args:
            mass: Mass in kg
            damping_coeff: Viscous damping coefficient in N·s/m (was 'friction')
            inertia: Moment of inertia in kg·m²
            radius: Characteristic radius in m (default 0.1m = 10cm)
        """
        self.mass = mass  # kg
        self.damping_coeff = damping_coeff  # N·s/m (viscous damping, not friction!)
        self.inertia = inertia  # kg·m²
        self.radius = radius  # m

        # Calculate effective mass for linear motion with rotational effects
        # m_eff = m + I/r² (dimensionally correct: kg + kg·m²/m² = kg)
        self.effective_mass = self.mass + (self.inertia / (self.radius ** 2))

        # State variables
        self.position = 0  # m
        self.velocity = 0  # m/s
        self.target = 1.0  # m (was 100, now realistic 1 meter)

        # Control limits (realistic actuator constraints)
        self.max_force = 50.0  # N
        self.min_force = -50.0  # N

    def reset(self):
        """Reset state to initial conditions."""
        self.position = 0
        self.velocity = 0

    def step(self, control_force, dt=0.01, with_saturation=True):
        """
        Simulate one timestep with proper physics.

        Args:
            control_force: Control force in Newtons
            dt: Time step in seconds
            with_saturation: Apply actuator limits

        Returns:
            Current position in meters
        """
        # Apply actuator saturation
        if with_saturation:
            control_force = np.clip(control_force, self.min_force, self.max_force)

        # Viscous damping force: F_damping = -b·v
        # Units: N = (N·s/m)·(m/s)
        damping_force = -self.damping_coeff * self.velocity

        # Net force
        net_force = control_force + damping_force

        # Newton's second law with effective mass
        # a = F/m_eff
        # Units: m/s² = N/kg
        acceleration = net_force / self.effective_mass

        # Euler integration
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position


def compute_iae(errors, dt=0.01):
    """Compute Integral Absolute Error."""
    return np.sum(np.abs(errors)) * dt


def compute_itae(errors, times, dt=0.01):
    """Compute Integral Time-weighted Absolute Error."""
    return np.sum(times * np.abs(errors)) * dt


def compute_ise(errors, dt=0.01):
    """Compute Integral Squared Error."""
    return np.sum(errors ** 2) * dt


def test_pid_with_antiwindup(robot, Kp, Ki, Kd, duration=5.0, dt=0.01,
                             noise_level=0.0, use_antiwindup=True,
                             integral_limit=10.0, use_d_filter=True,
                             d_filter_alpha=0.1):
    """
    Test PID controller with anti-windup and D-term filtering.

    Args:
        robot: Robot simulator instance
        Kp, Ki, Kd: PID gains
        duration: Simulation duration in seconds
        dt: Time step in seconds
        noise_level: Measurement noise standard deviation (fraction of target)
        use_antiwindup: Enable integral anti-windup
        integral_limit: Maximum integral term magnitude
        use_d_filter: Enable derivative filtering
        d_filter_alpha: Low-pass filter coefficient for D-term (0-1, lower = more filtering)

    Returns:
        Dictionary with performance metrics
    """
    robot.reset()

    times = np.arange(0, duration, dt)
    positions = []
    errors = []
    control_signals = []

    integral = 0
    prev_error = 0
    filtered_derivative = 0  # For D-term filtering

    # Performance tracking
    settling_time = duration
    max_overshoot = 0
    time_in_band = 0
    settling_threshold = 0.02 * robot.target  # 2% of target

    for i, t in enumerate(times):
        # Measure position with optional noise
        measured_position = robot.position
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * robot.target)
            measured_position += noise

        # Compute error
        error = robot.target - measured_position
        errors.append(error)

        # Proportional term
        p_term = Kp * error

        # Integral term with anti-windup
        integral += error * dt
        if use_antiwindup:
            # Clamp integral to prevent windup
            integral = np.clip(integral, -integral_limit, integral_limit)
        i_term = Ki * integral

        # Derivative term with optional filtering
        if i == 0:
            derivative = 0
        else:
            derivative = (error - prev_error) / dt

        if use_d_filter:
            # Low-pass filter for derivative
            filtered_derivative = (d_filter_alpha * derivative +
                                  (1 - d_filter_alpha) * filtered_derivative)
            d_term = Kd * filtered_derivative
        else:
            d_term = Kd * derivative

        # Total control signal
        control = p_term + i_term + d_term
        control_signals.append(control)

        # Apply control
        robot.step(control, dt)
        positions.append(robot.position)

        # Track overshoot
        if robot.position > robot.target:
            overshoot = (robot.position - robot.target) / robot.target  # Percentage
            max_overshoot = max(max_overshoot, overshoot)

        # Track settling time (within 2% band for 0.5 seconds)
        if abs(error) < settling_threshold:
            time_in_band += dt
            if time_in_band >= 0.5 and settling_time == duration:
                settling_time = t
        else:
            time_in_band = 0

        prev_error = error

    # Compute performance metrics
    errors_array = np.array(errors)
    times_array = np.array(times)

    # Standard control metrics
    iae = compute_iae(errors_array, dt)
    itae = compute_itae(errors_array, times_array, dt)
    ise = compute_ise(errors_array, dt)

    # Steady-state error
    steady_state_error = abs(errors[-1])

    # Rise time (0% to 90% of target)
    rise_time = duration
    for i, pos in enumerate(positions):
        if pos >= 0.9 * robot.target:
            rise_time = times[i]
            break

    return {
        'settling_time': settling_time,
        'overshoot': max_overshoot * 100,  # Percentage
        'ss_error': steady_state_error,
        'rise_time': rise_time,
        'iae': iae,
        'itae': itae,
        'ise': ise,
        'positions': positions,
        'errors': errors,
        'control_signals': control_signals,
        'times': times
    }


def get_adaptive_baseline_pid(mass, damping_coeff, inertia, radius=0.1):
    """
    Physics-based adaptive PID tuning with correct dimensions.

    Args:
        mass: Mass in kg
        damping_coeff: Damping coefficient in N·s/m
        inertia: Moment of inertia in kg·m²
        radius: Characteristic radius in m

    Returns:
        Tuple of (Kp, Ki, Kd) gains
    """
    # Calculate effective mass (dimensionally correct)
    eff_mass = mass + inertia / (radius ** 2)

    # Natural frequency estimate (rad/s)
    # Higher mass → lower frequency → slower response needed
    omega_n = np.sqrt(10.0 / eff_mass)  # Heuristic spring constant = 10 N/m

    # Damping ratio estimate
    # Higher damping → less overshoot
    zeta = damping_coeff / (2 * np.sqrt(10.0 * eff_mass))
    zeta = np.clip(zeta, 0.4, 1.2)  # Keep in reasonable range

    # PID gains based on desired closed-loop characteristics
    # These formulas are derived from second-order system analysis
    Kp = omega_n ** 2 * eff_mass  # Proportional to system stiffness
    Ki = 0.5 * Kp / (2 * zeta * omega_n)  # Integral for steady-state
    Kd = 2 * zeta * omega_n * eff_mass - damping_coeff  # Additional damping

    # Ensure positive gains
    Kp = max(0.1, Kp)
    Ki = max(0.0, Ki)
    Kd = max(0.0, Kd)

    return Kp, Ki, Kd


def get_cohen_coon_pid(mass, damping_coeff, inertia, radius=0.1):
    """
    Cohen-Coon tuning method adapted for our system.
    Based on first-order plus dead time (FOPDT) model approximation.
    """
    eff_mass = mass + inertia / (radius ** 2)

    # Approximate system as FOPDT
    K = 1.0 / damping_coeff  # Process gain
    tau = eff_mass / damping_coeff  # Time constant
    theta = 0.01  # Small dead time approximation

    # Cohen-Coon formulas
    ratio = theta / tau

    Kp = (1.35 / K) * (1 + 0.18 * ratio) / (1 - ratio)
    Ti = 2.5 * theta * (1 + 0.2 * ratio) / (1 - ratio)
    Td = 0.37 * theta * (1 / (1 - ratio))

    Ki = Kp / Ti if Ti > 0 else 0
    Kd = Kp * Td

    # Ensure reasonable bounds
    Kp = np.clip(Kp, 0.1, 50)
    Ki = np.clip(Ki, 0, 20)
    Kd = np.clip(Kd, 0, 10)

    return Kp, Ki, Kd


def get_chr_pid(mass, damping_coeff, inertia, radius=0.1, method='0_overshoot'):
    """
    Chien-Hrones-Reswick (CHR) tuning method.

    Args:
        method: '0_overshoot' or '20_overshoot' for different response types
    """
    eff_mass = mass + inertia / (radius ** 2)

    # System parameters
    K = 1.0 / damping_coeff
    tau = eff_mass / damping_coeff

    # CHR formulas (setpoint tracking)
    if method == '0_overshoot':
        # No overshoot
        Kp = 0.6 * tau / K
        Ti = 1.0 * tau
        Td = 0.5 * tau
    else:
        # 20% overshoot
        Kp = 0.95 * tau / K
        Ti = 1.35 * tau
        Td = 0.47 * tau

    Ki = Kp / Ti if Ti > 0 else 0
    Kd = Kp * Td

    # Bounds
    Kp = np.clip(Kp, 0.1, 50)
    Ki = np.clip(Ki, 0, 20)
    Kd = np.clip(Kd, 0, 10)

    return Kp, Ki, Kd


# Test the corrected simulator
if __name__ == "__main__":
    # Create robot with realistic parameters
    robot = RobotSimulator(
        mass=1.0,  # 1 kg
        damping_coeff=0.5,  # 0.5 N·s/m
        inertia=0.01,  # 0.01 kg·m²
        radius=0.1  # 10 cm
    )

    # Get baseline PID
    Kp, Ki, Kd = get_adaptive_baseline_pid(
        robot.mass, robot.damping_coeff, robot.inertia, robot.radius
    )

    print(f"Adaptive PID: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
    print(f"Effective mass: {robot.effective_mass:.3f} kg")

    # Test with anti-windup and filtering
    result = test_pid_with_antiwindup(
        robot, Kp, Ki, Kd,
        use_antiwindup=True,
        use_d_filter=True
    )

    print(f"\nPerformance Metrics:")
    print(f"Settling time: {result['settling_time']:.2f}s")
    print(f"Overshoot: {result['overshoot']:.1f}%")
    print(f"Rise time: {result['rise_time']:.2f}s")
    print(f"IAE: {result['iae']:.3f}")
    print(f"ITAE: {result['itae']:.3f}")
    print(f"ISE: {result['ise']:.3f}")

    # Plot response
    plt.figure(figsize=(12, 8))

    # Position response
    plt.subplot(2, 2, 1)
    plt.plot(result['times'], result['positions'], 'b-', label='Position')
    plt.axhline(y=robot.target, color='r', linestyle='--', label='Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('System Response')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Error
    plt.subplot(2, 2, 2)
    plt.plot(result['times'], result['errors'], 'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Tracking Error')
    plt.grid(True, alpha=0.3)

    # Control signal
    plt.subplot(2, 2, 3)
    plt.plot(result['times'], result['control_signals'], 'g-')
    plt.axhline(y=robot.max_force, color='r', linestyle=':', label='Saturation')
    plt.axhline(y=robot.min_force, color='r', linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Force (N)')
    plt.title('Control Signal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Phase portrait
    plt.subplot(2, 2, 4)
    velocities = np.diff(result['positions']) / 0.01
    plt.plot(result['positions'][:-1], velocities, 'b-', alpha=0.7)
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Phase Portrait')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TEST_PID_PNG, dpi=150)
    print(f"\nSaved plot to {TEST_PID_PNG}")