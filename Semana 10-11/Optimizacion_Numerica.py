"""
Computación Avanzada y Analítica
Semana: Optimización numérica

Implementaciones didácticas:
1. Búsqueda exhaustiva unidimensional.
2. Método de la sección áurea.
3. Diferenciación numérica mediante diferencias finitas centrales.
4. Gradiente numérico para funciones de varias variables.
5. Descenso por gradiente con gradiente analítico o numérico.
6. Ejemplos matemáticos y físicos.
7. Visualización de superficies, curvas de nivel y trayectorias.

Dependencias:
    numpy
    matplotlib

El archivo está pensado para ejecutarse por secciones en un IDE, Jupyter
Notebook o directamente desde la terminal.
"""

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Sequence[float]
ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]


# -----------------------------------------------------------------------------
# 1. OPTIMIZACIÓN UNIDIMENSIONAL
# -----------------------------------------------------------------------------

def exhaustive_search(
    f: ScalarFunction,
    a: float,
    b: float,
    n: int = 1000,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Busca aproximadamente el mínimo de f en [a,b] mediante una malla.

    Returns
    -------
    x_best, f_best, x_grid, y_grid
        Mejor punto encontrado, valor de la función y malla evaluada.
    """
    if b <= a:
        raise ValueError("Se requiere b > a.")
    if n < 2:
        raise ValueError("n debe ser al menos 2.")

    x_grid = np.linspace(a, b, n)
    y_grid = np.array([f(x) for x in x_grid], dtype=float)
    index = int(np.argmin(y_grid))

    return float(x_grid[index]), float(y_grid[index]), x_grid, y_grid


def golden_section_search(
    f: ScalarFunction,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[float, float, int]:
    """Minimiza una función unimodal en [a,b] con sección áurea."""
    if b <= a:
        raise ValueError("Se requiere b > a.")
    if tol <= 0:
        raise ValueError("tol debe ser positiva.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser positivo.")

    r = (math.sqrt(5.0) - 1.0) / 2.0

    x1 = b - r * (b - a)
    x2 = a + r * (b - a)
    f1 = float(f(x1))
    f2 = float(f(x2))

    iteration = 0

    while abs(b - a) > tol and iteration < max_iter:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - r * (b - a)
            f1 = float(f(x1))
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + r * (b - a)
            f2 = float(f(x2))

        iteration += 1

    x_best = (a + b) / 2.0
    return float(x_best), float(f(x_best)), iteration


def maximize_by_minimization(
    f: ScalarFunction,
    a: float,
    b: float,
    tol: float = 1e-6,
) -> Tuple[float, float, int]:
    """Maximiza f transformando el problema en minimizar -f."""
    x_best, negative_value, iterations = golden_section_search(
        lambda x: -f(x), a, b, tol=tol
    )
    return x_best, -negative_value, iterations


# -----------------------------------------------------------------------------
# 2. DIFERENCIACIÓN NUMÉRICA
# -----------------------------------------------------------------------------

def central_difference(
    f: ScalarFunction,
    x: float,
    h: float = 1e-5,
) -> float:
    """Aproxima f'(x) con diferencia central."""
    if h <= 0:
        raise ValueError("h debe ser positiva.")
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def numerical_gradient(
    f: VectorFunction,
    x: ArrayLike,
    h: float = 1e-5,
) -> np.ndarray:
    """Calcula el gradiente de f mediante diferencias finitas centrales."""
    if h <= 0:
        raise ValueError("h debe ser positiva.")

    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = h
        gradient[i] = (f(x + step) - f(x - step)) / (2.0 * h)

    return gradient


# -----------------------------------------------------------------------------
# 3. DESCENSO POR GRADIENTE
# -----------------------------------------------------------------------------

def gradient_descent(
    f: VectorFunction,
    gradient: GradientFunction,
    x0: ArrayLike,
    learning_rate: float = 0.1,
    tolerance: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, list, int]:
    """Minimiza f mediante descenso por gradiente.

    El criterio principal de parada es:
        ||x_(k+1) - x_k|| < tolerance

    Returns
    -------
    x : np.ndarray
        Último punto calculado.
    history : list
        Puntos visitados, incluyendo el inicial.
    iterations : int
        Número de actualizaciones realizadas.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positiva.")
    if tolerance <= 0:
        raise ValueError("tolerance debe ser positiva.")
    if max_iter <= 0:
        raise ValueError("max_iter debe ser positivo.")

    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]

    for iteration in range(1, max_iter + 1):
        grad = np.asarray(gradient(x), dtype=float)
        x_new = x - learning_rate * grad
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tolerance:
            return x_new, history, iteration

        x = x_new

    return x, history, max_iter


def gradient_descent_numerical(
    f: VectorFunction,
    x0: ArrayLike,
    learning_rate: float = 0.1,
    h: float = 1e-5,
    tolerance: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, list, int]:
    """Descenso por gradiente usando gradiente numérico."""
    return gradient_descent(
        f=f,
        gradient=lambda x: numerical_gradient(f, x, h=h),
        x0=x0,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_iter=max_iter,
    )


# -----------------------------------------------------------------------------
# 4. FUNCIONES DE EJEMPLO
# -----------------------------------------------------------------------------

def f_cuadratica_1d(x: float) -> float:
    """f(x) = (x-3)^2 + 2."""
    return (x - 3.0) ** 2 + 2.0


def energia_1d(x: float) -> float:
    """U(x) = x^4 - 4x^2 + 3."""
    return x**4 - 4.0 * x**2 + 3.0


def costo_diseno(x: float) -> float:
    """C(x) = 100/x + 2x^2."""
    return 100.0 / x + 2.0 * x**2


def funcion_2d(x: np.ndarray) -> float:
    """f(x,y) = x^2 + y^2."""
    return float(x[0] ** 2 + x[1] ** 2)


def gradiente_2d(x: np.ndarray) -> np.ndarray:
    """Gradiente de x^2+y^2."""
    return np.array([2.0 * x[0], 2.0 * x[1]])


def potencial_2d(x: np.ndarray) -> float:
    """U(x,y) = 1/4 (x^2-1)^2 + 1/2 y^2."""
    return float(0.25 * (x[0] ** 2 - 1.0) ** 2 + 0.5 * x[1] ** 2)


def gradiente_potencial_2d(x: np.ndarray) -> np.ndarray:
    """Gradiente analítico del potencial bidimensional."""
    return np.array([
        x[0] * (x[0] ** 2 - 1.0),
        x[1],
    ])


def energia_anillo(x: np.ndarray) -> float:
    """E(x,y) = (x^2+y^2-1)^2 + 0.2x^2."""
    r2 = x[0] ** 2 + x[1] ** 2
    return float((r2 - 1.0) ** 2 + 0.2 * x[0] ** 2)


def gradiente_energia_anillo(x: np.ndarray) -> np.ndarray:
    """Gradiente de la energía con término de penalización en x."""
    r2 = x[0] ** 2 + x[1] ** 2
    return np.array([
        4.0 * x[0] * (r2 - 1.0) + 0.4 * x[0],
        4.0 * x[1] * (r2 - 1.0),
    ])


# -----------------------------------------------------------------------------
# 5. VISUALIZACIÓN
# -----------------------------------------------------------------------------

def plot_1d_function(
    f: ScalarFunction,
    a: float,
    b: float,
    x_best: Optional[float] = None,
    title: str = "Función unidimensional",
) -> None:
    """Grafica una función de una variable y opcionalmente su óptimo."""
    x = np.linspace(a, b, 500)
    y = np.array([f(xi) for xi in x])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x)")
    if x_best is not None:
        plt.scatter([x_best], [f(x_best)], s=60, label="Óptimo")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_contours_with_paths(
    f: VectorFunction,
    paths: Iterable[Sequence[np.ndarray]],
    xlim: Tuple[float, float] = (-2.5, 2.5),
    ylim: Tuple[float, float] = (-2.5, 2.5),
    title: str = "Curvas de nivel y trayectorias",
) -> None:
    """Grafica curvas de nivel y una o varias trayectorias de optimización."""
    x = np.linspace(xlim[0], xlim[1], 250)
    y = np.linspace(ylim[0], ylim[1], 250)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30)

    for path in paths:
        path = np.asarray(path)
        plt.plot(path[:, 0], path[:, 1], marker="o", markersize=2)
        plt.scatter(path[0, 0], path[0, 1], s=50, marker="s")
        plt.scatter(path[-1, 0], path[-1, 1], s=50, marker="x")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_surface(
    f: VectorFunction,
    xlim: Tuple[float, float] = (-2.5, 2.5),
    ylim: Tuple[float, float] = (-2.5, 2.5),
    title: str = "Superficie de la función",
) -> None:
    """Grafica una superficie 3D para una función de dos variables."""
    x = np.linspace(xlim[0], xlim[1], 120)
    y = np.linspace(ylim[0], ylim[1], 120)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.75)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 6. DEMOSTRACIONES LISTAS PARA CLASE
# -----------------------------------------------------------------------------

def demo_unidimensional() -> None:
    """Ejecuta ejemplos de búsqueda exhaustiva y sección áurea."""
    print("--- Optimización unidimensional ---")

    x_grid, y_grid = None, None
    x_exh, y_exh, x_grid, y_grid = exhaustive_search(
        f_cuadratica_1d, 0.0, 6.0, n=1000
    )
    print(f"Búsqueda exhaustiva: x = {x_exh:.8f}, f(x) = {y_exh:.8f}")

    x_gs, y_gs, iterations = golden_section_search(
        f_cuadratica_1d, 0.0, 6.0, tol=1e-8
    )
    print(
        f"Sección áurea:       x = {x_gs:.8f}, "
        f"f(x) = {y_gs:.8f}, iteraciones = {iterations}"
    )

    x_cost, y_cost, _ = golden_section_search(
        costo_diseno, 1.0, 10.0, tol=1e-8
    )
    print(
        f"Costo de diseño:     x = {x_cost:.8f}, "
        f"C(x) = {y_cost:.8f}"
    )

    plot_1d_function(
        f_cuadratica_1d,
        0.0,
        6.0,
        x_best=x_gs,
        title="Minimización de (x-3)^2 + 2",
    )


def demo_descenso_por_gradiente() -> None:
    """Ejecuta descenso por gradiente en una función cuadrática."""
    print("--- Descenso por gradiente ---")

    x_min, history, iterations = gradient_descent(
        f=funcion_2d,
        gradient=gradiente_2d,
        x0=[4.0, 3.0],
        learning_rate=0.1,
        tolerance=1e-8,
        max_iter=1000,
    )

    print("Mínimo aproximado:", x_min)
    print("Valor de la función:", funcion_2d(x_min))
    print("Iteraciones:", iterations)

    plot_contours_with_paths(
        funcion_2d,
        [history],
        xlim=(-4, 4),
        ylim=(-4, 4),
        title="Descenso por gradiente: x^2 + y^2",
    )


def demo_potencial_fisico() -> None:
    """Estudia diferentes condiciones iniciales para un potencial físico."""
    print("--- Aplicación física: potencial bidimensional ---")

    initial_points = [
        [2.0, 1.0],
        [-2.0, 1.0],
        [0.5, 1.5],
        [-0.5, -1.5],
    ]

    paths = []

    for x0 in initial_points:
        x_min, history, iterations = gradient_descent(
            f=potencial_2d,
            gradient=gradiente_potencial_2d,
            x0=x0,
            learning_rate=0.15,
            tolerance=1e-8,
            max_iter=2000,
        )
        paths.append(history)
        print(
            f"Inicio {x0}: mínimo = {x_min}, "
            f"U = {potencial_2d(x_min):.8f}, "
            f"iteraciones = {iterations}"
        )

    plot_contours_with_paths(
        potencial_2d,
        paths,
        xlim=(-2.2, 2.2),
        ylim=(-2.2, 2.2),
        title="Paisaje de energía y trayectorias",
    )

    plot_surface(
        potencial_2d,
        xlim=(-2.0, 2.0),
        ylim=(-2.0, 2.0),
        title="Superficie de energía potencial",
    )


def demo_gradiente_numerico() -> None:
    """Compara gradiente analítico y numérico."""
    print("--- Comparación de gradientes ---")
    points = [
        np.array([1.2, -0.8]),
        np.array([-1.5, 0.6]),
        np.array([0.3, 1.7]),
    ]

    for point in points:
        analytic = gradiente_potencial_2d(point)
        numeric = numerical_gradient(potencial_2d, point, h=1e-5)
        error = np.linalg.norm(analytic - numeric)

        print(f"Punto: {point}")
        print(f"  Gradiente analítico: {analytic}")
        print(f"  Gradiente numérico:  {numeric}")
        print(f"  Error: {error:.3e}")


def demo_varios_learning_rates() -> None:
    """Compara diferentes tasas de aprendizaje en una función cuadrática."""
    print("--- Efecto de la tasa de aprendizaje ---")

    learning_rates = [0.01, 0.1, 0.5, 0.9]

    for alpha in learning_rates:
        x_min, history, iterations = gradient_descent(
            f=funcion_2d,
            gradient=gradiente_2d,
            x0=[4.0, 3.0],
            learning_rate=alpha,
            tolerance=1e-8,
            max_iter=100,
        )
        print(
            f"alpha={alpha:>4}: "
            f"punto={x_min}, "
            f"f={funcion_2d(x_min):.3e}, "
            f"iteraciones={iterations}"
        )


# -----------------------------------------------------------------------------
# 7. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("OPTIMIZACIÓN NUMÉRICA - COMPUTACIÓN AVANZADA Y ANALÍTICA")
    print("=" * 70)

    demo_unidimensional()
    demo_descenso_por_gradiente()
    demo_gradiente_numerico()
    demo_varios_learning_rates()
    demo_potencial_fisico()
