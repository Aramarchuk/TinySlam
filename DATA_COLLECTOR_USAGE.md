# Data Collector Documentation

## Overview

`DataCollector` - это модуль для сбора и хранения всех данных симуляции SLAM. Он собирает полную информацию о процессе симуляции для последующего анализа и вычисления метрик.

## Features

### Собираемые данные:
- **Траектории**: Реальные и оцененные позиции робота на каждом шаге
- **Карты**: Состояние карты робота с заданной частотой
- **Данные сенсоров**: Показания лидара или наблюдения ориентиров
- **Метаданные**: Информация о параметрах симуляции

### Типы данных:
- **Grid-based SLAM** (TinySLAM): Данные лидара, сеточные карты
- **Landmark-based SLAM** (EKF): Наблюдения ориентиров, пустые карты

## Usage

### Basic Usage

```python
from data_collector import SimulationDataCollector

# Создание коллектора
collector = SimulationDataCollector(simulation_name="MySLAM")

# Настройка параметров
collector.set_ground_truth_map(ground_truth_map)
collector.set_simulation_info(
    total_steps=300,
    slam_algorithm="TinySLAM",
    map_type="grid"
)
collector.set_map_frequency(10)  # Сохранять карту каждые 10 шагов

# Сбор данных в цикле симуляции
for step in range(total_steps):
    # ... выполнение шага симуляции ...

    collector.add_frame(
        step=step,
        ground_truth_pose=(x_gt, y_gt, theta_gt),
        estimated_pose=(x_est, y_est, theta_est),
        robot_map=robot_map,
        sensor_data=sensor_measurements
    )

# Сохранение данных
saved_files = collector.save_to_file("output_directory")
```

### Integration in main.py

Модуль уже интегрирован в `main.py` и работает автоматически:

```bash
# Запуск симуляции с автоматическим сбором данных
python main.py --slam ts  # TinySLAM
python main.py --slam ek  # EKF SLAM
```

## Data Structure

### Основные атрибуты:
- `ground_truth_poses`: List[(x, y, theta)] - реальные позиции
- `estimated_poses`: List[(x, y, theta)] - оцененные позиции
- `robot_maps`: List[{'step': int, 'map': np.array}] - карты робота
- `sensor_data`: List[{'step': int, 'data': sensor_data}] - данные сенсоров
- `timestamps`: List[int] - номера шагов
- `ground_truth_map`: np.array - реальная карта мира

### Метаданные:
- `simulation_name`: Название симуляции
- `slam_algorithm`: Используемый алгоритм
- `map_type`: Тип карты ("grid" или "landmark")
- `total_steps`: Общее количество шагов
- `map_size`: Размер карты (N, N)
- `collected_frames`: Количество собранных кадров

## File Format

### Сохраняемые файлы:
1. **`*_main.pkl`** - Основные данные (траектории, метаданные)
2. **`*_maps.pkl`** - Карты робота (отдельно для экономии памяти)
3. **`*_metadata.json`** - Метаданные в JSON для удобного просмотра
4. **`*_report.txt`** - Текстовый отчет о собранных данных

### Пример структуры файлов:
```
sim_steps/run_20251105_140406/
├── TinySLAM_20251105_140432_main.pkl      # Основные данные
├── TinySLAM_20251105_140432_maps.pkl      # Карты
├── TinySLAM_20251105_140432_metadata.json # Метаданные
├── TinySLAM_20251105_140432_report.txt   # Отчет
└── ...
```

## Data Analysis

### Загрузка данных:
```python
from data_collector import SimulationDataCollector

# Загрузка сохраненных данных
collector = SimulationDataCollector()
collector.load_from_file("path/to/main.pkl", "path/to/maps.pkl")

# Анализ траекторий
trajectory_data = collector.get_trajectory_data()
pose_errors = collector.get_pose_errors()

# Статистика ошибок
position_errors = [error['position_error'] for error in pose_errors]
print(f"Average position error: {np.mean(position_errors):.4f}")
```

### Встроенные методы анализа:
- `get_trajectory_data()`: Получение данных траекторий
- `get_pose_errors()`: Вычисление ошибок позиций
- `print_summary()`: Вывод сводной информации

## Testing

### Тестирование загрузки и анализа:
```bash
# Запуск тестирования на последних данных
python test_data_collector.py
```

Скрипт автоматически:
1. Находит последние сохраненные данные
2. Загружает их
3. Проводит полный анализ
4. Создает визуализацию траекторий
5. Генерирует отчет

## Configuration Options

### Настройка частоты сохранения:
```python
# Сохранять карту каждый шаг (много памяти)
collector.set_map_frequency(1)

# Сохранять карту каждые 10 шагов (баланс)
collector.set_map_frequency(10)

# Сохранять карту каждые 50 шагов (экономия памяти)
collector.set_map_frequency(50)
```

### Управление памятью:
- Карты занимают больше всего памяти
- Рекомендуется `map_frequency=10` для 300 шагов
- Для больших симуляций увеличивайте частоту

## Examples

### Пример 1: Сбор базовой статистики
```python
collector = SimulationDataCollector("test")
# ... сбор данных ...

pose_errors = collector.get_pose_errors()
position_errors = [e['position_error'] for e in pose_errors]

print(f"Mean error: {np.mean(position_errors):.3f}")
print(f"Max error: {np.max(position_errors):.3f}")
print(f"Std error: {np.std(position_errors):.3f}")
```

### Пример 2: Сравнение алгоритмов
```python
# Запускаем обе симуляции
os.system("python main.py --slam ts")
os.system("python main.py --slam ek")

# Загружаем и сравниваем результаты
ts_collector = load_latest_data("TinySLAM")
ekf_collector = load_latest_data("EkfStubSLAM")

ts_errors = [e['position_error'] for e in ts_collector.get_pose_errors()]
ekf_errors = [e['position_error'] for e in ekf_collector.get_pose_errors()]

print(f"TinySLAM mean error: {np.mean(ts_errors):.3f}")
print(f"EKF mean error: {np.mean(ekf_errors):.3f}")
```

## Performance Considerations

### Оптимизация памяти:
- Используйте `set_map_frequency()` для редкого сохранения карт
- Для больших симуляций рассмотрите сохранение только траекторий

### Оптимизация скорости:
- Сбор данных добавляет ~1-2ms на шаг
- Сохранение больших файлов может занимать время

### Рекомендуемые настройки:
- **TinySLAM**: `map_frequency=10` (31 карта для 300 шагов)
- **EKF SLAM**: `map_frequency=1` (карты пустые, можно чаще)

## Troubleshooting

### Common Issues:

1. **Memory Error**:
   - Увеличьте `map_frequency`
   - Уменьшите количество шагов

2. **File Not Found**:
   - Проверьте пути к файлам
   - Убедитесь, что симуляция завершилась успешно

3. **Loading Error**:
   - Проверьте версии Python и библиотек
   - Убедитесь, что файлы не повреждены

### Debug Information:
```python
# Проверка собранных данных
collector.print_summary()

# Проверка конкретных данных
print(f"Trajectory points: {len(collector.ground_truth_poses)}")
print(f"Saved maps: {len(collector.robot_maps)}")
print(f"Sensor records: {len(collector.sensor_data)}")
```

## Future Extensions

Планируемые улучшения:
- Поддержка HDF5 формата для больших данных
- Потоковое сохранение в реальном времени
- Сжатие данных для экономии памяти
- Встроенная визуализация в реальном времени