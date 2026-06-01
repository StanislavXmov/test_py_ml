import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_LABELS = [
    "Площадь, м2",
    "Комнат",
    "Этаж",
    "Возраст дома, лет",
    "До центра, км",
]


def generate_apartment_dataset(
    n_samples: int = 500, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Синтетические данные: цена зависит от площади, комнат, этажа, возраста дома и удаленности от центра."""
    rng = np.random.default_rng(seed)

    area = rng.integers(25, 121, size=n_samples)
    rooms = rng.integers(1, 5, size=n_samples)
    floor = rng.integers(1, 26, size=n_samples)
    building_age = rng.integers(0, 51, size=n_samples)
    distance_km = rng.uniform(1, 25, size=n_samples)

    # Цена в млн руб.: больше площадь и комнат -> дороже; старый дом и далеко от центра -> дешевле
    price_mln = (
        2.0
        + 0.09 * area
        + 1.1 * rooms
        + 0.03 * floor
        - 0.04 * building_age
        - 0.18 * distance_km
        + rng.normal(0, 0.4, size=n_samples)
    )
    price_mln = np.clip(price_mln, 3.0, None)

    x = np.column_stack([area, rooms, floor, building_age, distance_km])
    return x, price_mln


def format_price_mln(price_mln: float) -> str:
    return f"{price_mln:.2f} млн руб."


def main() -> None:
    x, y = generate_apartment_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("=== Прогноз цены квартир (LinearRegression) ===\n")
    print("Признаки модели:")
    for label in FEATURE_LABELS:
        print(f"  - {label}")
    print()
    print("Влияние признаков (млн руб. на единицу):")
    for label, coef in zip(FEATURE_LABELS, model.coef_):
        sign = "+" if coef >= 0 else ""
        print(f"  {label:22s}: {sign}{coef:.3f}")
    print(f"  {'Базовая цена':22s}: {model.intercept_:.3f}")
    print()
    print(f"Качество на тесте, R2: {r2_score(y_test, y_pred):.3f}")
    print(f"Средняя ошибка (MAE): {mean_absolute_error(y_test, y_pred):.2f} млн руб.")
    print()

    examples = np.array(
        [
            [45, 2, 5, 10, 3.0],
            [72, 3, 12, 5, 8.0],
            [95, 4, 18, 35, 20.0],
        ]
    )
    example_labels = [
        "Небольшая, близко к центру",
        "Средняя квартира",
        "Большая квартира на окраине",
    ]

    print("Примеры прогноза:")
    for label, row, pred in zip(example_labels, examples, model.predict(examples)):
        area, rooms, floor, age, dist = row
        print(f"  {label}:")
        print(
            f"    {int(area)} м2, {int(rooms)} комн., этаж {int(floor)}, "
            f"дом {int(age)} лет, {dist:.1f} км от центра"
        )
        print(f"    -> {format_price_mln(pred)}")
        print()

    print("Сравнение факт vs прогноз (5 квартир из теста):")
    for i in range(5):
        actual = y_test[i]
        predicted = y_pred[i]
        print(
            f"  факт: {format_price_mln(actual)}, "
            f"прогноз: {format_price_mln(predicted)}"
        )


# x, y = generate_apartment_dataset(4)
# print(x)
# print(y)

main()
