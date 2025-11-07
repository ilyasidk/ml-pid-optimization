import numpy as np
import matplotlib.pyplot as plt
from config import TEST_PID_PNG

class RobotSimulator:
    """Базовая физическая модель робота"""
    
    def __init__(self, mass=1.0, friction=0.5, inertia=0.1):
        self.mass = mass
        self.friction = friction
        self.inertia = inertia
        
        # State variables
        self.position = 0
        self.velocity = 0
        self.target = 100
        
    def reset(self):
        """Сброс состояния"""
        self.position = 0
        self.velocity = 0
        
    def step(self, control_force, dt=0.01):
        """Один шаг симуляции (F=ma + rotational inertia effect)"""
        # Friction force
        friction_force = -self.friction * self.velocity

        # Net force
        net_force = control_force + friction_force

        # Effective mass includes rotational inertia effect
        # For combined translation+rotation: m_eff = m + I/r^2 (assuming r=1)
        effective_mass = self.mass + self.inertia

        # Newton's second law: a = F/m_eff
        acceleration = net_force / effective_mass

        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position


def test_pid(robot, Kp, Ki, Kd, duration=5.0, dt=0.01, noise_level=0.0):
    """
    Тестирует PID контроллер на роботе
    Args:
        robot: робот для тестирования
        Kp, Ki, Kd: PID параметры
        duration: длительность симуляции
        dt: шаг времени
        noise_level: уровень шума в измерениях позиции (0.0 = без шума)
    Returns: metrics (settling_time, overshoot, error, score)
    """
    robot.reset()
    
    times = np.arange(0, duration, dt)
    positions = []
    
    integral = 0
    prev_error = 0
    
    settling_time = duration
    max_overshoot = 0
    time_in_band = 0
    
    for t in times:
        # Measure position with noise
        measured_position = robot.position
        if noise_level > 0:
            measured_position += np.random.normal(0, noise_level * robot.target)

        # PID calculation using noisy measurement
        error = robot.target - measured_position
        integral += error * dt
        derivative = (error - prev_error) / dt

        control = Kp * error + Ki * integral + Kd * derivative

        # Apply control
        robot.step(control, dt)
        positions.append(robot.position)
        
        # Track overshoot
        if robot.position > robot.target:
            overshoot = robot.position - robot.target
            max_overshoot = max(max_overshoot, overshoot)
        
        # Track settling time (within 2% for 0.5 sec)
        if abs(error) < 2:
            time_in_band += dt
            if time_in_band >= 0.5 and settling_time == duration:
                settling_time = t
        else:
            time_in_band = 0
        
        prev_error = error
    
    # Final metrics
    steady_state_error = abs(robot.target - robot.position)
    
    # Performance score (lower = better)
    score = settling_time + max_overshoot * 2 + steady_state_error * 5
    
    return {
        'settling_time': settling_time,
        'overshoot': max_overshoot,
        'ss_error': steady_state_error,
        'score': score,
        'positions': positions
    }


def get_adaptive_baseline_pid(mass, friction, inertia):
    """
    Адаптивный baseline на основе физических параметров робота
    Эвристика, основанная на физической интуиции:
    - Высокая масса → lower Kp (избежать колебаний)
    - Высокое трение → higher Ki (компенсировать steady-state error)
    - Высокая инерция → higher Kd (демпфировать overshoot)
    """
    # Эффективная масса (включая инерцию)
    eff_mass = mass + inertia

    # Адаптивные коэффициенты
    Kp = 3.0 / (eff_mass ** 0.5)  # Inversely proportional to sqrt(mass)
    Ki = 0.5 + 1.5 * friction / (eff_mass)  # Higher for high friction
    Kd = 0.3 * inertia * (eff_mass ** 0.3)  # Higher for high inertia

    return Kp, Ki, Kd


def get_ziegler_nichols_pid(robot, test_duration=10.0, dt=0.01):
    """
    Упрощенный метод Ziegler-Nichols (Ultimate Gain Method)
    Находит Ku (критический коэффициент) и Tu (период колебаний)

    Note: Это упрощенная имитация, т.к. для полного ZN требуется
    реальный эксперимент с увеличением Kp до критического значения
    """
    # Поиск критического Kp (где система начинает колебаться)
    robot.reset()

    # Binary search для Ku
    Kp_low, Kp_high = 0.1, 50.0
    Ku = None
    Tu = None

    for _ in range(10):  # Максимум 10 итераций
        Kp_test = (Kp_low + Kp_high) / 2
        robot.reset()

        # Тест с только P-контроллером
        positions = []
        times = np.arange(0, test_duration, dt)
        integral = 0
        prev_error = 0

        for t in times:
            error = robot.target - robot.position
            control = Kp_test * error  # Только P
            robot.step(control, dt)
            positions.append(robot.position)

        # Проверка на sustained oscillations
        pos_array = np.array(positions[-200:])  # Последние 2 секунды
        mean_pos = np.mean(pos_array)
        std_pos = np.std(pos_array)

        # Если колеблется (std достаточно большое)
        if std_pos > 5.0:
            Ku = Kp_test
            # Оценка периода колебаний (упрощенно)
            Tu = 2.0  # Примерная оценка
            break
        elif mean_pos < robot.target * 0.8:
            # Не дошли - нужен больший Kp
            Kp_low = Kp_test
        else:
            # Перескочили - нужен меньший Kp
            Kp_high = Kp_test

    # Если не нашли Ku, используем эвристику
    if Ku is None:
        Ku = 5.0
        Tu = 2.0

    # Ziegler-Nichols формулы для PID
    Kp = 0.6 * Ku
    Ki = 2.0 * Kp / Tu
    Kd = Kp * Tu / 8.0

    return Kp, Ki, Kd


# TEST: проверь что работает
if __name__ == "__main__":
    robot = RobotSimulator(mass=1.0, friction=0.5, inertia=0.1)
    result = test_pid(robot, Kp=2.0, Ki=0.5, Kd=0.3)
    
    print(f"Settling time: {result['settling_time']:.2f}s")
    print(f"Overshoot: {result['overshoot']:.2f}")
    print(f"Score: {result['score']:.2f}")
    
    # Plot
    plt.plot(result['positions'])
    plt.axhline(y=100, color='r', linestyle='--', label='Target')
    plt.xlabel('Time steps')
    plt.ylabel('Position')
    plt.title('PID Response')
    plt.legend()
    plt.savefig(TEST_PID_PNG)
    print(f"Saved plot to {TEST_PID_PNG.name}")