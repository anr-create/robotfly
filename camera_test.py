import cv2
import numpy as np


# --- 1. КЛАСС НЕЙРОНА LIF ---
class LIFNeuron:
    """
    Класс Leaky Integrate-and-Fire (LIF) нейрона.
    Моделирует работу биоинспирированного нейрона с накоплением заряда,
    утечкой, порогом срабатывания и рефрактерным периодом (паузой для отдыха).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        leak: float = 0.1,
        refractory_ticks: int = 2
    ):
        self.threshold = threshold          # Порог срабатывания (выдача спайка 1)
        self.leak = leak                    # Утечка заряда за 1 такт
        self.refractory_ticks = refractory_ticks  # Длительность отдыха (в тактах)
        
        self.charge = 0.0                   # Текущий мембранный потенциал (заряд)
        self.refractory_counter = 0         # Счетчик оставшихся тактов отдыха

    @property
    def is_ready(self) -> bool:
        """Булевый флаг: готов ли нейрон принимать входящие сигналы."""
        return self.refractory_counter == 0

    def update(self, input_signal: float) -> int:
        """
        Обновляет состояние нейрона на каждом такте цикла while True.
        
        :param input_signal: Входящий сигнал (сенсоры OpenCV + шум)
        :return: 1 если сгенерирован спайк (выстрел), иначе 0
        """
        # 1. Если нейрон отдыхает — уменьшаем счетчик и игнорируем входящий сигнал
        if not self.is_ready:
            self.refractory_counter -= 1
            return 0

        # 2. Утечка заряда (заряд угасает со временем, не опускается ниже 0.0)
        self.charge = max(0.0, self.charge - self.leak)

        # 3. Интегрирование (накопление) входящего сигнала
        self.charge += input_signal

        # 4. Проверка порога и генерация спайка
        if self.charge >= self.threshold:
            self.charge = 0.0                               # Сброс заряда
            self.refractory_counter = self.refractory_ticks # Запуск рефрактерной паузы
            return 1                                        # ⚡ Спайк!

        return 0  # Спайка нет


# --- 2. ГЛАВНАЯ ПРОГРАММА (ИНТЕГРАЦИЯ С КАМЕРОЙ) ---
def main():
    # Подключаемся к веб-камере (0 — встроенная камера)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Ошибка: Не удалось открыть веб-камеру!")
        raise SystemExit

    print("✅ Камера открыта. Нажмите 'q' в окне видео, чтобы выйти.")

    # Создаем два независимых нейрона для левого и правого глаза
    left_neuron = LIFNeuron(threshold=0.8, leak=0.08, refractory_ticks=2)
    right_neuron = LIFNeuron(threshold=0.8, leak=0.08, refractory_ticks=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break

        # Сжимаем кадр до 160x120 пикселей для высокой производительности
        small_frame = cv2.resize(frame, (160, 120))

        # Переводим в черно-белый формат (Grayscale)
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Делим кадр ровно пополам (левый и правый сектора зрения)
        left_sector = gray_frame[:, :80]
        right_sector = gray_frame[:, 80:]

        # Считываем уровень света от 0.0 до 1.0
        left_brightness = np.mean(left_sector) / 255.0
        right_brightness = np.mean(right_sector) / 255.0

        # Передаем яркость в нейроны и получаем спайки (выстрелы)
        left_spike = left_neuron.update(left_brightness)
        right_spike = right_neuron.update(right_brightness)

        # Рисуем интерфейс отладки на исходном кадре
        h, w, _ = frame.shape
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

        # Вывод заряда и статуса левого нейрона
        left_status = "SPIKE!" if left_spike else f"{left_neuron.charge:.2f}"
        cv2.putText(
            frame, f"Left Eye Charge: {left_status}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not left_spike else (0, 0, 255), 2
        )

        # Вывод заряда и статуса правого нейрона
        right_status = "SPIKE!" if right_spike else f"{right_neuron.charge:.2f}"
        cv2.putText(
            frame, f"Right Eye Charge: {right_status}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not right_spike else (0, 0, 255), 2
        )

        # Показываем окно с видео
        cv2.imshow("RoboFly Brain & Vision", frame)

        # Нажатие клавиши 'q' закрывает программу
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()