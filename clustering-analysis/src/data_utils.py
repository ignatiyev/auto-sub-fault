"""
    Вспомогательные функции для упрощения работы с файлами (в основном ASCII)
    и ввода-вывода данных, оценки плотности, сглаживания 2D Гауссом и т.д.
"""
import os
import numpy as np
import scipy.io
#================================================================================
#                           Ввод-вывод данных
#================================================================================   
def removeColumn(file_in, lCol):
    """
    Удаление всех столбцов, указанных в lCol
    1) Создаёт дубликат файла с названием 'dummy_file.txt' в текущей директории
    2) Удаляет столбцы с помощью awk
    3) Возвращает имя файла дубликата
    """
    # Пример синтаксиса для удаления трёх столбцов
    # os.system("awk '{$24=""; $25=""; $26=""; print}' in_file.txt > out_file.txt")
    lStr = []
    for col in lCol:
        lStr.append("$%s=\"\"; "%(col))
    tmp_file = 'dummy_file.txt'
    command_str = "awk '{ %s print}' %s > %s"%( ''.join(lStr), file_in, tmp_file)
    os.system(command_str)           
    return tmp_file

def loadmat(filename, verbose=False):
    '''
    Эта функция должна использоваться вместо прямого вызова scipy.io.loadmat,
    который используется внутри метода
        (1) - фильтрует теги словаря
        (2) - корректно восстанавливает словари Python
              из файлов mat. Проверяет теги dic, которые всё ещё являются mat-объектами
        (3) - исправляет массивы вида: np.array([[ 1, 2, 3]]) в np.array([ 1, 2, 3]), squeeze_me=True
        (4) - может обрабатывать 'вложенные' переменные в Matlab, где переменная содержит несколько структур
    
    '''
    data = scipy.io.loadmat(filename, struct_as_record=True, squeeze_me=True)
    data = _check_keys(data)
    for tag in list(data.keys()):
        if tag[0] == '_':
            if verbose == True:
                print('Удаляем', tag, data[tag])
            data.pop(tag)
    return data

def _check_keys(dData):
    '''
    Проверяет, являются ли элементы в словаре mat-объектами. Если да,
    вызывается to_dict для их преобразования в вложенные словари
    '''
    for key in dData:
        if isinstance(dData[key], scipy.io.matlab.mio5_params.mat_struct):
            dData[key] = _todict(dData[key])
    return dData        

def _todict(matobj):
    '''
    Рекурсивная функция, которая создаёт вложенные словари из mat-объектов
    '''
    dData = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dData[strg] = _todict(elem)
        else:
            dData[strg] = elem
    return dData
#================================================================================
#                           Оценка плотности и сглаживание
#================================================================================   
def density_2D(x, y, x_bin, y_bin, **kwargs):
    """
        2D, сглаженная плотность событий для облака точек с координатами x,y
        использует метод: scipy.stats.kde.gaussian_kde
        см.: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    :входные данные     x,y            - набор данных
               x_bin, y_bin   - векторы x и y в виде бинов
    
        kwargs['sigma'] - указывает ядро сглаживания Гаусса ('bw_method' в scipy.stats.kde)
                          по умолчанию: = n**( -1./(d+3)) адаптированное правило Скотта для немного более узкой полосы
                        - 'scott' 
                              sigma = n**( -1./(d+4)), d - число измерений, n - число точек данных
                        - 'silverman'
                              sigma = (n * (d + 2) / 4.)**(-1. / (d + 4))
                        - float( )   = устанавливает ширину полосы Гаусса напрямую
                                        
                           
    возвращает XX, YY, ZZ - 2D биннированные координаты x и y и плотность для каждой ячейки
    """
    from scipy.stats import kde
    n, d = x.shape[0], 2
    sigma = n ** (-1. / (d + 2.5))
    if 'sigma' in kwargs.keys() and kwargs['sigma'] is not None:
        sigma = kwargs['sigma']
    # Оценка гауссовского KDE на регулярной сетке nbins x nbins в пределах данных
    fct_Gauss2D = kde.gaussian_kde(np.array([x, y]), bw_method=sigma)
    # Сетка координат x и y
    XX, YY = np.meshgrid(x_bin, y_bin)
    ZZ = fct_Gauss2D(np.vstack([XX.flatten(), YY.flatten()])).reshape(XX.shape)
    dx, dy = x_bin[1] - x_bin[0], y_bin[1] - y_bin[0]
    # Проверка, является ли интеграл ~ нулю, лучше: использовать метод середины
    print('Проверка, является ли интеграл ~1', round(ZZ.sum() * (dx * dy), 3)) # ZZ[ZZ>0].mean()*(XX.max()-XX.min())*(YY.max()-YY.min()))
    return XX - .5 * dx, YY - .5 * dy, ZZ

#================================================================================
#                          Обработка словарей
#================================================================================ 
def copyDic(dic):
    """ Создаёт копию словаря """
    import copy
    dCopy = {}
    for tag in dic.keys():
        dCopy[tag] = copy.copy(dic[tag])
    return dCopy

def selectDataRange(dicOri, min, max, tag, **kwargs):
    """
    Выбирает данные в заданном диапазоне, установите min = None или max = None для только нижней или верхней границы
    """
    dic = copyDic(dicOri)
    if 'includeBoundaryEvents' in kwargs.keys() and kwargs['includeBoundaryEvents'] == True:
        if min == None or max == None:
            error_str = 'Обе границы должны быть установлены для включения граничных событий'
            raise ValueError(error_str)
        else:
            sel = np.logical_and(dic[tag] >= float(min), dic[tag] <= float(max))          
    if max == None:
        sel = dic[tag] > float(min)
    elif min == None:
        sel = dic[tag] < max
    else:
        sel = np.logical_and(dic[tag] > float(min), dic[tag] < float(max))
    sel = np.arange(dic[tag].shape[0], dtype=int)[sel]
    if 'returnSel' in kwargs.keys() and kwargs['returnSel'] == True:
        return sel
    else:        
        return selDicAll(dic, sel, **kwargs)

def selDicAll(dic, curr_sel, **kwargs):
    """Применяет булев вектор ко всем данным
    например, для сортировки или обрезки ... """
    newDic = {}
    for tag, vector in dic.items():
        newDic[tag] = dic[tag][curr_sel]
    return newDic