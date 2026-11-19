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


# --- БЛОК ТЕСТИРОВАНИЯ ---
if __name__ == "__main__":
    # Создаем объект нейрона с порогом 1.0, утечкой 0.1 и отдыхом на 2 такта
    neuron = LIFNeuron(threshold=1.0, leak=0.1, refractory_ticks=2)

    print("--- Запуск теста нейрона ---")
    
    # Подаем постоянный сигнал 0.4 на протяжении 12 тактов
    
    for step in range(1, 13):
        spike = neuron.update(input_signal=0.4)
        status = "ГОТОВ" if neuron.is_ready else f"ОТДЫХ ({neuron.refractory_counter})"
        
        print(
            f"Такт {step:2d} | "
            f"Заряд: {neuron.charge:.2f} | "
            f"Статус: {status:10s} | "
            f"Спайк: {spike}"
        )