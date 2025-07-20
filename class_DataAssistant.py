# Основной класс ассистента (DataAssistant)
class DataAssistant:
    # Конструктор класса
    def __init__(self):
        self.df = None          # Переменная для хранения датасета
        self.X = None           # Признаки модели
        self.y = None           # Целевые значения
        self.scaled_X = None    # Масштабированные признаки
        self.models = {}        # Словарь для хранения моделей

    # Метод загрузки данных
    def load_data(self, filename=None):
        # Если имя файла не передано в качестве аргумента, запрашиваем его у пользователя
        if not filename:
            filename = input("Введите путь к файлу с данными (.csv): ")
        try:
            # Пытаемся загрузить данные из CSV-файла в DataFrame с помощью pandas
            self.df = pd.read_csv(filename)
            print("Датасет успешно загружен!\n")
        except FileNotFoundError:
            # Если файл не найден, выводим сообщение об ошибке
            print("Файл не найден.")

    # Метод для вывода базовой статистики данных
    def show_stats(self):
        # Проверяем, загружены ли данные в DataFrame
        if self.df is None:
            # Если данные не загружены, возвращаем сообщение об ошибке
            return "Сначала загрузите данные!"

        # Выводим базовую статистику по данным в DataFrame
        # Метод describe() возвращает такие показатели, как среднее, стандартное отклонение, минимальное и максимальное значения и квартильные значения
        print(self.df.describe())

    # Метод для визуализации данных
    def visualize_data(self):
        # Проверяем, загружены ли данные
        if self.df is None:
            return "Сначала загрузите данные!"

        # Получаем список столбцов с числовыми данными
        numeric_columns = list(self.df.select_dtypes(include=['number']).columns)

        # Создаем парные графики для числовых столбцов
        # Используем последний столбец в качестве параметра 'hue' для цветовой дифференциации
        sns.pairplot(self.df, vars=numeric_columns, hue=self.df.columns[-1])

        # Отображаем графики
        plt.show()

    # Добавлю метод предварительной обработки данных
    def preprocess_data(self):
        # Проверяем, загружены ли данные. Если нет, возвращаем сообщение об ошибке.
        if self.df is None:
            return "Сначала загрузите данные!"

        # Применяем one-hot encoding для всех категориальных столбцов.
        # Сначала определяем категориальные столбцы, исключая числовые типы данных.
        categorical_cols = self.df.select_dtypes(exclude=["float64", "int64"]).columns

        # Преобразуем категориальные столбцы в one-hot encoding, удаляя первый уровень, чтобы избежать дамми-ловушки.
        df_encoded = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

        # Заполняем пропущенные значения в числовых столбцах средним значением.
        imputer = SimpleImputer(strategy='mean')
        numeric_cols = df_encoded.select_dtypes(include=['float64', 'int64'])
        df_encoded[numeric_cols.columns] = imputer.fit_transform(numeric_cols)

        # Масштабируем данные, чтобы все числовые признаки имели среднее значение 0 и стандартное отклонение 1.
        scaler = StandardScaler()
        self.scaled_X = scaler.fit_transform(df_encoded.iloc[:, :-1])

        # Отделяем целевую переменную (предполагается, что она находится в последнем столбце).
        self.y = df_encoded.iloc[:, -1]

        # Выводим сообщение о том, что данные успешно подготовлены.
        print("Данные подготовлены!")

    # Метод построения и оценки моделей
    def build_models(self):
        # Проверяем, были ли предварительно обработаны данные
        if self.scaled_X is None or self.y is None:
            return "Сначала выполните препроцессинг данных!"

        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_X, self.y, test_size=0.3, random_state=42)

        # Создаем экземпляры моделей: Дерево решений и Логистическая регрессия
        dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
        lrc = LogisticRegression(random_state=42)

        # Обучаем модели на обучающей выборке
        dtc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # Сохраняем обученные модели в словарь для дальнейшего использования
        self.models['Decision Tree'] = dtc
        self.models['Logistic Regression'] = lrc

        # Оцениваем каждую модель на тестовой выборке
        for name, model in self.models.items():
            # Получаем предсказания модели
            predictions = model.predict(X_test)

            # Генерируем отчет о классификации
            report = classification_report(y_test, predictions)

            # Создаем матрицу ошибок (confusion matrix)
            matrix = confusion_matrix(y_test, predictions)

            # Выводим отчет о классификации и матрицу ошибок
            print(f"\n{name} Classification Report:\n{report}\nConfusion Matrix:\n{matrix}")

    # Интерфейс командной строки для управления ассистентом
    def run(self):
        commands = {     # Добавляем словарь команд прямо сюда
            'load': self.load_data,
            'stats': self.show_stats,
            'visualize': self.visualize_data,
            'preprocess': self.preprocess_data,
            'build': self.build_models,
            'exit': sys.exit
        }

        while True:
            cmd = input("\nВведите команду ('help' для справки): ").strip().lower()

            if cmd == 'help':
                print("Доступные команды:\n"
                      "- load: Загрузить датасет\n"
                      "- stats: Показать статистику\n"
                      "- visualize: Визуализировать данные\n"
                      "- preprocess: Преобразовать и нормализовать данные\n"
                      "- build: Построить и оценить модели\n"
                      "- exit: Завершить работу ассистента")

            elif cmd in commands:
                commands[cmd]()  # Выполнение соответствующей команды
                plt.pause(0.1)  # Задержка для обновления графиков



            else:
                print("Неверная команда. Используйте 'help'.")
