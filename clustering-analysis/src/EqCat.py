"""Каталоги землетрясений класса "Анализ сейсмических каталогов" - данные хранятся в словаре, который можно легко расширять, при условии, что все векторы имеют одинаковую длину. Базовые функции сосредоточены на вводе/выводе каталога и первичной обработке (пространство, время, выбор окна магнитуд)"""

import copy
import os
import numpy as np 
import src.datetime_utils as dateTime
import scipy.io  # Модуль для чтения и записи .mat файлов (бинарный формат MATLAB)

class EqCat:
    """
    Класс для работы с каталогами землетрясений.
    (1) self.data - словарь Python для хранения данных каталога.
    Пример структуры словаря:
    self.data = { 
        'N': np.array([]),  # Номер события (ID)
        'Time': np.array([]),  # Время в десятичных годах
        'Lon': np.array([]),  # Долгота (lon)
        'Lat': np.array([]),  # Широта (lat)
        'Depth': np.array([]),  # Глубина (depth)
        'Mag': np.array([]),  # Магнитуда
    }
    """

    def __init__(self, **kwargs):
        """Инициализация объекта класса EqCat"""

        # Создание пустого словаря для хранения данных каталога
        self.data = {}

        # Создаем список всех публичных методов объекта (не начинаются с '_')
        # dir(self) возвращает все атрибуты и методы объекта
        # getattr(self, method_name) получает объект метода по его имени
        # callable() проверяет, является ли объект вызываемым (функция/метод)
        self.methods = [method_name for method_name in dir(self)
                        if callable(getattr(self, method_name)) and method_name[0] != '_']

        # Установка имен полей для координат (по умолчанию GPS-координаты: долгота, широта, глубина)
        # Если передан аргумент 'type' в kwargs, можно переключиться на декартовы координаты (X, Y, Z)
        self.sLoc1, self.sLoc2, self.sLoc3 = 'Lon', 'Lat', 'Depth'
        # Установка имени поля для ID события
        self.sID = 'N'

    def copy(self, catalog):
        """Создание глубокой копии каталога"""
        try:
            # Копируем данные из переданного каталога в self.data
            # Используем copy.copy для каждого вектора данных
            for tag, vector in catalog.data.items():
                self.data[tag] = copy.copy(catalog.data[tag])
        except:
            # Если catalog.data недоступен, пробуем копировать напрямую из catalog
            for tag, vector in catalog.items():
                self.data[tag] = copy.copy(catalog[tag])

    #===========================================================================
    #                         Импорт каталогов
    #===========================================================================
    def loadEqCat(self, file_in, catalogType, verbose=False, **kwargs):
        """
        Загрузка каталога землетрясений из файла.
        Входные параметры:
        - file_in: имя файла каталога
        - catalogType: тип каталога ('USGS' и т.д.)
        - verbose: если True, выводит прогресс обработки
        - kwargs['header']: номер строки с заголовками столбцов
        - kwargs['removeColumn']: номера столбцов для удаления перед загрузкой
        Возвращает: объект каталога с данными в self.data = {'Time', 'Lon', 'Lat', 'Depth', 'Mag'}
        """
        # Проверка аргумента header в kwargs
        if 'header' in kwargs.keys() and kwargs['header'] is not None:
            header = kwargs['header']
        else:
            header = None

        # Если указаны столбцы для удаления, удаляем их из файла и создаем временную копию
        if 'removeColumn' in kwargs.keys() and kwargs['removeColumn'] is not None:
            import src.data_utils as data_utils
            file_in = data_utils.removeColumn(file_in, kwargs['removeColumn'])

        # Обработка разных типов каталогов
        if catalogType == 'USGS':
            # Каталог USGS: содержит данные в формате CSV с определенными столбцами
            # Загружаем дату и время (столбцы 0, 2, 4, 6, 8, 10) с разделителями
            mDateTime = np.genfromtxt(file_in, delimiter=(4,1,2,1,2,1,2,1,2,1,4),
                                      skip_header=1, usecols=(0,2,4,6,8,10)).T
            headDate = ['YR', 'MO', 'DY', 'HR', 'MN', 'SC']
            # Заполняем словарь данными о времени
            for i in range(len(headDate)):
                self.data[headDate[i]] = mDateTime[i]
            # Создаем ID для каждого события (простая нумерация)
            self.data['ID'] = np.arange(len(self.data['YR']))
            # Загружаем координаты и магнитуду (столбцы 1, 2, 3, 4)
            header = ['Lat', 'Lon', 'Depth', 'Mag']
            mData = np.loadtxt(file_in, delimiter=',', skiprows=1, usecols=(1,2,3,4), dtype=float).T
            # Заполняем словарь данными о координатах и магнитуде
            for i in range(len(header)):
                self.data[header[i]] = mData[i]

        # Преобразуем дату и время в десятичные годы
        self.data['Time'] = np.array([])
        for i in range(self.data['Mag'].shape[0]):
            if verbose:
                print(i+1, 'out of', self.data['Mag'].shape[0])  # Прогресс обработки
            # Проверяем корректность даты и времени
            YR, MO, DY, HR, MN, SC = dateTime.checkDateTime([self.data['YR'][i], self.data['MO'][i], self.data['DY'][i], self.data['HR'][i], self.data['MN'][i], self.data['SC'][i]])
            # Преобразуем дату в десятичные годы и добавляем в массив
            self.data['Time'] = np.append(self.data['Time'], 
                                          dateTime.dateTime2decYr([YR, MO, DY, HR, MN, SC]))
        # Сортируем каталог по времени
        self.sortCatalog('Time')

        # Очистка: удаляем временный файл, если он был создан
        if 'removeColumn' in kwargs.keys() and kwargs['removeColumn'] is not None:
            print("delete: %s, than hit: y" % (file_in))
            removeFile = input(' ')
            if os.path.isfile(file_in) and removeFile == 'y':
                os.system("rm %s" % (file_in))

    #======================================2==========================================
    #                            Базовая обработка и выбор событий
    #=================================================================================
    def size(self):
        """Возвращает количество событий в каталоге (длина массива 'Time')"""
        if 'Time' in self.data.keys():
            return len(self.data['Time'])
        else:
            return None

    def selectEvents(self, min, max, tag, **kwargs):
        """
        Выбирает события, соответствующие заданному диапазону значений по тегу (например, 'Time' или 'Mag').
        Входные параметры:
        - min, max: границы диапазона (нижняя и верхняя)
        - tag: ключ словаря, по которому выполняется фильтрация ('Time', 'Mag' и т.д.)
        - kwargs['includeBoundaryEvents']: включать ли граничные события (min и max)
        - kwargs['returnSel']: вернуть массив индексов вместо изменения каталога
        Пример: selectEvents(3, 5, 'Mag', includeBoundaryEvents=True) - события с магнитудой от 3 до 5
        """
        if 'includeBoundaryEvents' in kwargs.keys() and kwargs['includeBoundaryEvents']:
            if min is None or max is None:
                raise ValueError('both boundaries have to be set to include boundary events')
            else:
                # Выбираем события, где значение тега находится в диапазоне [min, max]
                sel = np.logical_and(self.data[tag] >= float(min), self.data[tag] <= float(max))
        else:
            if isinstance(min, str):
                # Если min - строка, фильтруем по точному совпадению (например, для 'magType')
                sel = [i for i, x in enumerate(self.data[tag]) if x == min]
            elif isinstance(min, (int, float)) or min is None:
                if max is None:
                    # Выбираем события, где значение тега >= min
                    sel = self.data[tag] >= float(min)
                elif min is None:
                    # Выбираем события, где значение тега < max
                    sel = self.data[tag] < max
                else:
                    # Выбираем события в диапазоне [min, max)
                    sel = np.logical_and(self.data[tag] >= float(min), self.data[tag] < float(max))
            else:
                raise ValueError('unknown input min = %s' % (min))
        if 'returnSel' in kwargs.keys() and kwargs['returnSel']:
            return sel  # Возвращаем массив индексов
        else:
            self.selDicAll(sel)  # Применяем выборку ко всем данным

    def sortCatalog(self, tag, **kwargs):
        """Сортировка каталога по заданному тегу (например, 'Time', 'Mag')"""
        # Получаем индексы для сортировки
        vSortBool = self.data[tag].ravel().argsort()
        if 'beginWithBiggest' in kwargs.keys() and kwargs['beginWithBiggest']:
            if 'returnSel' in kwargs.keys() and kwargs['returnSel']:
                return vSortBool[::-1]  # Возвращаем индексы в обратном порядке (от большего к меньшему)
            else:
                self.selDicAll(vSortBool[::-1])  # Сортируем от большего к меньшему
        else:
            if 'returnSel' in kwargs.keys() and kwargs['returnSel']:
                return vSortBool  # Возвращаем индексы в порядке возрастания
            else:
                self.selDicAll(vSortBool)  # Сортируем по возрастанию

    def selDicAll(self, sel):
        """Применяет булев массив или индексы ко всем данным в словаре"""
        for tag, vector in self.data.items():
            # Применяем выборку (sel) к каждому вектору данных
            self.data[tag] = self.data[tag][sel]

    def selEventsFromID(self, a_ID, **kwargs):
        """
        Выбирает события по списку ID (self.data['N']).
        Входные параметры:
        - a_ID: массив ID событий
        - kwargs['repeats']: если True, сохраняет повторяющиеся ID в порядке их появления
        """
        Nev = len(a_ID)
        repeats = False
        if 'repeats' in kwargs.keys() and kwargs['repeats']:
            a_sel = np.ones(Nev, dtype=int)
            v_i = np.arange(self.size(), dtype=int)
            i = 0
            # Для каждого ID ищем первое совпадение и сохраняем его индекс
            for currID in a_ID:
                sel_curr_ev = self.data['N'] == int(currID)
                if sel_curr_ev.sum() > 0:
                    a_sel[i] = int(v_i[sel_curr_ev][0])
                i += 1
        else:
            # Выбираем события, где ID присутствует в a_ID (без повторов)
            a_sel = np.in1d(self.data['N'], a_ID, assume_unique=True)
        self.selDicAll(a_sel)

    #======================================3==========================================
    #                            Работа с .mat файлами
    #=================================================================================
    def check_keys(self):
        """Проверяет, есть ли в словаре структуры MATLAB, и преобразует их в словари Python"""
        for key in self.data:
            if isinstance(self.data[key], scipy.io.matlab.mio5_params.mat_struct):
                self.data[key] = self.todict(self.data[key])

    def todict(self, matobj):
        """Рекурсивно преобразует MATLAB-структуры в словари Python"""
        dData = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                dData[strg] = self.todict(elem)  # Рекурсивно обрабатываем вложенные структуры
            else:
                dData[strg] = elem
        return dData

    def saveMatBin(self, file):
        """Сохраняет словарь self.data в .mat файл (бинарный формат MATLAB)"""
        scipy.io.savemat(file, self.data, appendmat=True, format='5', do_compression=True)

    def loadMatBin(self, filename):
        """
        Загружает данные из .mat файла, обрабатывая вложенные структуры.
        Используется вместо прямого вызова scipy.io.loadmat для корректной работы с MATLAB-структурами.
        """
        self.data = {}
        # Загружаем данные из .mat файла
        self.data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        self.check_keys()  # Обрабатываем вложенные структуры
        # Удаляем системные ключи (начинаются с '_')
        l_tags = list(self.data.keys())
        for tag in l_tags:
            if tag[0] == '_':
                self.data.pop(tag, None)

    #======================================4==========================================
    #                            Проекции и преобразования координат
    #=================================================================================
    def toCart_coordinates(self, **kwargs):
        """
        Преобразует GPS-координаты (Lon, Lat) в декартовы (X, Y) с использованием проекции.
        Входные параметры:
        - kwargs['projection']: тип проекции ('aeqd' - азимутальная равноудаленная, 'eqdc', 'cyl')
        - kwargs['returnProjection']: если True, возвращает объект Basemap
        Выход: добавляет в self.data поля 'X', 'Y' (в км) и сохраняет 'Depth'
        """
        # Установка пути к PROJ_LIB (необходимо для Basemap, закомментировать, если не нужно)
        os.environ["PROJ_LIB"] = f"{os.environ['HOME']}/opt/anaconda3/share/proj"
        from mpl_toolkits.basemap import Basemap
        projection = 'aeqd'  # Проекция по умолчанию
        if 'projection' in kwargs.keys() and kwargs['projection'] is not None:
            projection = kwargs['projection']
        # Определяем границы области по долготе и широте
        xmin, xmax = self.data['Lon'].min(), self.data['Lon'].max()
        ymin, ymax = self.data['Lat'].min(), self.data['Lat'].max()
        # Создаем объект Basemap для преобразования координат
        m = Basemap(llcrnrlat=ymin, urcrnrlat=ymax,
                    llcrnrlon=xmin, urcrnrlon=xmax,
                    projection=projection, lat_0=(ymin+ymax)*0.5, lon_0=(xmin+xmax)*0.5,
                    resolution='l')
        # Преобразуем долготу и широту в X, Y
        self.data['X'], self.data['Y'] = m(self.data['Lon'], self.data['Lat'])
        if projection != 'cyl':
            # Преобразуем координаты из метров в километры
            self.data['X'] *= 1e-3
            self.data['Y'] *= 1e-3
        if 'returnProjection' in kwargs.keys() and kwargs['returnProjection']:
            return m  # Возвращаем объект Basemap
        else:
            return True

    #======================================5==========================================
    #                            Перемешивание и случайные каталоги
    #=================================================================================
    def randomize_cat(self):
        """
        Создает случайный каталог с той же средней частотой, количеством событий 
        и пространственным распределением, что и исходный каталог.
        Возвращает: случайный каталог с равномерным пространственным распределением
        """
        ## Перемешивание времени событий (реализация не завершена в предоставленном коде)