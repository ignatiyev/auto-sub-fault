'''
Функции, необходимые для анализа кластеризации на основе расстояний до ближайшего соседа
'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
#===============================================================================
#                          Собственные модули
#===============================================================================
import src.data_utils as data_utils

#===============================================================================
# 
#===============================================================================
def NND_eta( eqCat, dConst, verbose = False, **kwargs):
    """
        - NND_eta - уравнение 1 для NND в Zaliapin & Ben-Zion 2013
    Поиск "родительского события", т.е. землетрясения, ближайшего в пространственно-временном-магнитудном домене, 
    произошедшего ранее текущего события
        здесь: [jC]          - дочерние события, для которых мы ищем ближайшее родительское событие, произошедшее ранее
              [sel_tau_par] - потенциальные родительские события, произошедшие до [jC], выбираем ближайшее по времени

    Параметры
    ----------
    eqCat     - каталог данных: catalog.data['Time'], 'Lon', 'Lat' (или 'X', 'Y'), 'Depth', 'Mag'
                - время, декартовы координаты (X, Y, Depth), магнитуда
    dConst    - {'Mc':float, 'b':float, 'D':float} # словарь со статистическими параметрами сейсмичности
                - пороговая магнитуда, b-значение, фрактальная размерность
    kwargs    - rmax (по умолчанию: = 500) - максимальное пространственное окно (для ускорения вычислений)
              - tmax (по умолчанию: =  20) - максимальное временное окно (для ускорения вычислений)
              - correct_co_located = True, добавить гауссову погрешность, чтобы избежать нуля для совпадающих землетрясений
              - haversine = True - использовать расстояние по формуле гаверсинуса вместо 3D декартова расстояния
              - M0 - опорная магнитуда, по умолчанию: M0 = 0

    Возвращает
    -------
    - {  'aNND'       : aNND,     - расстояние до ближайшего соседа в пространственно-временном-магнитудном домене
         'aEqID_p'    : np.array  - ID родительского события
         'aEqID_c'    : np.array  - ID дочернего события
         'Time'       : np.array  - время возникновения дочерних событий
        } 

    См.: Clustering Analysis of Seismicity and Aftershock Identification, Zaliapin, I. (2008)
    """
    #-------------------------------set args and kwargs----------------------------------------------- 
    rmax = 500 # в километрах
    tmax = 20 # в годах
    M0 = 0 # опорная магнитуда
    if 'M0' in kwargs.keys() and kwargs['M0'] is not None:
        M0 = kwargs['M0']
    if 'rmax' in kwargs.keys() and kwargs['rmax'] is not None:
        rmax = kwargs['rmax']
    if 'tmax' in kwargs.keys() and kwargs['tmax'] is not None:
        tmax = kwargs['tmax']
    #-----------------------------add small uncertainty to X in case events are colocated-------------------------- 
    if 'correct_co_located' in kwargs.keys() and kwargs['correct_co_located'] == True:
        vUncer = np.random.randn( eqCat.size())*1e-10
        eqCat.data['Lon']    += vUncer
    #------------------------------------------------------------------------------
    aNND     = np.zeros( eqCat.size())
    vID_p    = np.zeros( eqCat.size())
    vID_c    = np.zeros( eqCat.size())
    a_M_MS_ref= (eqCat.data['Mag'] - M0)# магнитуда главного толчка относительно опорной
 
    for jC in range( eqCat.size()):
        if verbose == True:
            print( f"Событие {jC+1:d} из {eqCat.size():d}", end= "\r")
        # interevent times: take events that happend before t_i 
        #           child             - parent                > 0 
        tau         =  eqCat.data['Time'][jC] - eqCat.data['Time']
        sel_tau_par = tau > 0
        if sel_tau_par.sum() > 0:

            vcurr_ID = np.arange( eqCat.size(), dtype = int)[sel_tau_par]
            # if cartesian coordinates are available
            if 'X' in eqCat.data.keys() and 'Y' in eqCat.data.keys():
                vR = np.sqrt( (eqCat.data['X'][jC] - eqCat.data['X'][vcurr_ID])**2 + (eqCat.data['Y'][jC] - eqCat.data['Y'][vcurr_ID])**2 )
            else:
                # расстояние по формуле гаверсинуса
                vR = haversine( eqCat.data['Lon'][jC], eqCat.data['Lat'][jC],eqCat.data['Lon'][vcurr_ID], eqCat.data['Lat'][vcurr_ID] )
            sel_r_par = vR < rmax
            if sel_r_par.sum() > 0:
                vcurr_ID = vcurr_ID[sel_r_par]
                curr_Eta = tau[vcurr_ID]* (vR[sel_r_par]**dConst['D']) *( 10**(-dConst['b']*a_M_MS_ref[vcurr_ID]))
                sel_min  = curr_Eta == curr_Eta.min()
                aNND[jC]    = curr_Eta[sel_min][0]
                vID_p[jC]   = eqCat.data['N'][vcurr_ID][sel_min][0]
                vID_c[jC]   = eqCat.data['N'][jC]
                #print( 'родитель', eqCat.data['N'][vcurr_ID][sel_min][0],  'потомок', eqCat.data['N'][jC]
                #print( 'родитель', eqCat.data['Time'][vcurr_ID][sel_min][0],  'потомок', eqCat.data['Time'][jC]

                if sel_min.sum() > 1:
                    print( aNND[jC], curr_Eta[sel_min], eqCat.data['N'][vcurr_ID][sel_min])
                    print( eqCat.data['Lon'][vcurr_ID][sel_min], eqCat.data['Lat'][vcurr_ID][sel_min])
    sel2 = aNND > 0
    if np.logical_not(sel2).sum() > 0:
        pass
        # print( f"{np.logical_not(sel2).sum()} %i событий с NND=0 ")
        #raise ValueError, error_str
    # удаляем события с aNND < 0, т.е. события в начале без предшествующего родителя
    return {  'aNND' : aNND[sel2], 'aEqID_p' : vID_p[sel2], 'aEqID_c' : vID_c[sel2], 'Time' : eqCat.data['Time'][sel2]}
    #return {  'aNND' : aNND, 'aEqID_p' : vID_p, 'aEqID_c' : vID_c, 'Time' : eqCat.data['Time'][1::]}


def rFromTau( dt, b, D, eta_0, M_MS ):
    """
        - Вычисление максимального расстояния R для событий в кластере
          на основе межсобытийного времени, eta_0 и фрактальной размерности D
    :Входные данные
          dt    - массив или число
               межсобытийные времена (dt относительно главного толчка или первого события в кластере)
          b     - b-значение по Гутенбергу-Рихтеру
          D     - фрактальная размерность, обычно D~1.6
          eta_0 - эмпирически определённая линия разделения между кластерным и фоновым режимами
          M_MS  - магнитуда главного толчка (здесь предполагается только одна триггерная генерация)
    :Возвращает:
          расстояние R
    """
    return ( -eta_0/dt * 10**( b*M_MS))**(1/D)*1e-3

def rescaled_t_r(catChild, catPar, dConst, **kwargs):
    """
    - Вычисление масштабированного времени и расстояния

    Параметры
    ----------
    catChild, catPar - объекты типа SeisCatDic, содержащие родительские и дочерние события
    dConst      =  'b', 'D' -  b-значение, фрактальная размерность
    kwargs       = distance_3D = True по умолчанию: False, т.е. 2D евклидово расстояние

    Возвращает
    -------
    - a_R, a_tau


    См.: Clustering Analysis of Seismicity and Aftershock Identification, Zaliapin, I. (2008)
    """
    #-------------------------------set args and kwargs-----------------------------------------------
    M0 = 0
    if 'M0' in kwargs.keys() and kwargs['M0'] is not None:
        M0 = kwargs['M0']
    #-----------------------------add small uncertainty to X in case events are colocated-------------------------- 
    if 'correct_co_located' in kwargs.keys() and kwargs['correct_co_located'] == True:
        vUncer = np.random.randn( catChild.size())*1e-10
        catChild.data['Lon']    += vUncer
    #------------------------------------------------------------------------------         
    #vMagCorr = 10**(-0.5*dConst['b']*(catPar.data['MAG']-M0) )
    vMagCorr = 10**(-0.5*dConst['b']*(catPar.data['Mag']-M0) )
    # if cartesian coordinates are available
    if 'X' in catChild.data.keys() and 'X' in catPar.data.keys():
        a_R = np.sqrt((catChild.data['X'] - catPar.data['X']) ** 2 + (catChild.data['Y'] - catPar.data['Y']) ** 2) ** \
              dConst['D'] * vMagCorr

    else:
        a_R = haversine(catChild.data['Lon'], catChild.data['Lat'],
                   catPar.data['Lon'],   catPar.data['Lat'])**dConst['D']*vMagCorr

    a_dt = catChild.data['Time']-catPar.data['Time']#межсобытийные времена
    a_tau = (a_dt)*vMagCorr
    sel2 = a_tau < 0
    if sel2.sum() > 0:
        #print( catChild.data['N'][sel2])
        #print( catPar.data['N'][sel2])
        error_str = '%i родительских событий произошли после дочерних, проверьте порядок времени в catChild, catPar'%(sel2.sum())
        raise( ValueError( error_str))
    return a_R, a_tau


def compileClust( dNND, simThreshold, verbose = True,  **kwargs):
    """
    Предполагается, что родитель и потомок связаны уникальным измерением (например, расстоянием до ближайшего соседа)
    - Создание кластеров пар событий на основе критерия схожести
            например: a) на основе коэффициентов кросс-корреляции между парами
                     b) на основе пространственно-временного-магнитудного расстояния
    - Основной вход: пары связанных событий, разделённые на родительские и дочерние
                  (один родитель может иметь много потомков, но у потомка только один родитель)
    1) Найти одиночные события за пределами порога
    2) Найти пары ниже порога и собрать кластеры
        - Взять все пары событий с значениями ниже (eta_0) или выше (CCC),
           --> пары за пределами порога не рассматриваются
            если потомок удовлетворяет критерию схожести:
                - пройти по каждой паре и найти кластер для дочернего события, проверяя,
                  есть ли соответствующий ID в предыдущих кластерах
                - присоединить к существующему кластеру или создать новый
    3) - Проверить, связаны ли несколько потомков с одним родителем, и объединить кластеры
         при повторении ID
         --> реализовано как цикл while
    4) - Удалить возможные множественные ID из кластеров

    :Входные данные    - simThreshold = параметр схожести
              - vID_parent   - ID событий
              - vID_child
              - vSimValues   - все значения схожести
              kwargs['useLargerEvents'] = False, 

    :Возвращает  dClust - словарь Python, содержащий все кластеры, пронумерованные
                     от '0' - не кластеризованные
                          '1' - '[nCLmax]' - кластеризованные события
                      каждая колонка словаря содержит ID дочерних [первая строка] и родительских [вторая строка] событий
    """
    # dNND = { 'aEqID_c' : vID_child,
    #          'aEqID_p' : vID_parent,
    #          'aNND'    : vSim}
    # удаляем одинаковых родителей и потомков, если событие встречается в каталоге несколько раз
    sel = abs(dNND['aEqID_c']-dNND['aEqID_p']) > 0
    dNND= data_utils.selDicAll(dNND, sel)

    # проверяем, что dNND отсортирован по времени
    if 'Time' in dNND.keys():
        i_sort = np.argsort( dNND['Time'])
        dNND   = data_utils.selDicAll(dNND, i_sort)
    else:
        error_str = "Отсутствует ключ 'Time', добавьте время возникновения дочерних событий в dNND"
        raise ValueError( error_str)
    #==================================1=============================================
    #                  начальная выборка событий за пределами порога (одиночные события)
    #================================================================================
    ### события без триггера
    if 'useLargerEvents' in kwargs.keys() and kwargs['useLargerEvents'] == True:
        print( 'Предполагается, что порог (%s) — это МИНИМУМ, выбираем значения схожести ВЫШЕ этого порога'%( simThreshold))
        sel_single     = dNND['aNND'] <= simThreshold
        # удаляем независимые события
        dNND_trig = data_utils.selectDataRange( dNND, simThreshold, None, 'aNND')
    else:
        print( 'Предполагается, что порог (%s) — это МАКСИМУМ, выбираем значения схожести НИЖЕ этого порога'%( simThreshold))
        sel_single     = dNND['aNND'] >= simThreshold
        # удаляем независимые события
        dNND_trig = data_utils.selDicAll( dNND, np.logical_not( sel_single))
    # предварительная выборка одиночных событий с eta > eta_0, может содержать кластерные события
    vID_single  = dNND['aEqID_c'][sel_single] # могут быть одиночными или родительскими, но не дочерними
    sel_first = np.in1d( dNND['aEqID_p'][0], vID_single)
    if dNND['aNND'][0] > simThreshold and sel_first.sum() == 0:
        vID_single = np.append(  dNND['aEqID_p'][0], vID_single)

    if verbose == True:
        print(f"---------compileClust - начальные числа:------")
        print(f"Количество одиночных: {vID_single.shape[0]}"),
        print(f"Количество триггерных: {dNND_trig['aEqID_c'].shape[0]}, {dNND_trig['aEqID_p'].shape[0]},"),
        print(f"Общее количество: {dNND_trig['aEqID_p'].shape[0]} {sel_single.sum()+dNND_trig['aEqID_c'].shape[0]}")
    #==================================2=============================================
    #                      поиск кластеризованных событий
    #================================================================================
    # инициализация векторов и словаря при первом запуске
    curr_child_ID     = dNND_trig['aEqID_c'][0]
    curr_par_ID       = dNND_trig['aEqID_p'][0]
    v_pastEqIDs = np.array(  [curr_child_ID, curr_par_ID] )
    v_pastClIDs = np.array(  [1, 1] )
    # dClust['0'] = одиночные
    dClust = {  '1'     : np.array( [[curr_child_ID],
                                     [curr_par_ID  ] ])}
    # для каждого дочернего события найти соответствующий ID родителя
    # если ID дочернего или родительского события уже есть в кластере, добавить к этому кластеру
    nCl = 2
    for iEv in range(1, dNND_trig['aEqID_p'].shape[0]):
        #print( 'nPair', iEv+1, 'из', len( dNND_trig['aEqID_p']), 'iCl', nCl
        curr_child_ID     = dNND_trig['aEqID_c'][iEv]
        curr_par_ID       = dNND_trig['aEqID_p'][iEv]
        # проверяем, есть ли родитель или потомок в предыдущем кластере
        sel_child = curr_child_ID == v_pastEqIDs
        sel_par   = curr_par_ID   == v_pastEqIDs

        if sel_par.sum() > 0 or sel_child.sum() > 0:
            # определяем, к какому кластеру относится пара событий
            if sel_par.sum() and sel_child.sum(): # оба уже в кластере
                curr_cl_ID1 = v_pastClIDs[sel_par][0]
                curr_cl_ID2 = v_pastClIDs[sel_child][0]
                # объединяем кластеры и добавляем ID
                dClust[str(curr_cl_ID1)] =    np.hstack( (dClust[str(curr_cl_ID1)],
                                                             np.array([[curr_child_ID], [curr_par_ID  ] ])
                                                             ))
                dClust[str(curr_cl_ID1)] = np.hstack( (dClust[str(curr_cl_ID1)], dClust[str(curr_cl_ID2)]))
                # добавляем новые события, но сохраняем предыдущий ID кластера
                v_pastEqIDs = np.append(  v_pastEqIDs, np.array([curr_child_ID, curr_par_ID] ) )
                v_pastClIDs = np.append(  v_pastClIDs, np.array([   curr_cl_ID1, curr_cl_ID1] ) )
                # удаляем второй ID кластера из dClust
                dClust.pop( str(curr_cl_ID2))
                # удаляем из предыдущих ID событий и ID кластеров
                sel = curr_cl_ID2 != v_pastClIDs
                v_pastEqIDs = v_pastEqIDs[sel]
                v_pastClIDs = v_pastClIDs[sel]
            else: # только один в кластере
                if sel_par.sum() > 0: # родитель уже в кластере
                    curr_cl_ID = v_pastClIDs[sel_par][0]
                else:# потомок уже в кластере
                    curr_cl_ID = v_pastClIDs[sel_child][0]
                dClust[str(curr_cl_ID)] =    np.hstack( (dClust[str(curr_cl_ID)],
                                                             np.array([[curr_child_ID], [curr_par_ID  ] ])
                                                             ))
                v_pastEqIDs = np.append(  v_pastEqIDs, np.array([curr_child_ID, curr_par_ID] ) )
                v_pastClIDs = np.append(  v_pastClIDs, np.array([   curr_cl_ID, curr_cl_ID        ] ) )
        else: # начинаем новый кластер
            dClust[str(nCl)] =    np.array( [[curr_child_ID],
                                             [curr_par_ID  ] ])
            v_pastEqIDs = np.append(  v_pastEqIDs, np.array([curr_child_ID, curr_par_ID] ) )
            v_pastClIDs = np.append(  v_pastClIDs, np.array([          nCl, nCl        ] ) )
            nCl += 1
    # проверяем, есть ли у потомков один и тот же родитель
    nTotChild = 0
    #=================================3==========================================================================
    #                 удаляем события из одиночных, если они в кластере, удаляем множественные ID
    #============================================================================================================
    # создаём вектор ID триггерных событий и подсчитываем их количество
    vID_Trig_all = np.array([])
    vclID_allEv  = np.array([], dtype = int)
    for tag in sorted( dClust.keys()):
        #print( 'iCl', tag, 'количество событий в кластере', np.unique( dClust[tag].flatten()).shape[0]
        #print( dClust[tag][0]
        aID_flat_uni = np.unique( dClust[tag].flatten())
        #nTotTrig   += aID_flat_uni.shape[0]
        vID_Trig_all = np.append( vID_Trig_all, aID_flat_uni )
        vclID_allEv  = np.append( vclID_allEv, np.ones( aID_flat_uni.shape[0], dtype = int)*int(tag))
        # удаляем множественные записи ID --> возможно, так как пары всегда добавляются
        dClust[tag] = aID_flat_uni
        nTotChild  += dClust[tag].shape[0]-1
    #====================================4========================================================================
    #                       проверяем события, находящиеся в нескольких кластерах, объединяем кластеры
    #============================================================================================================
    # sel_same = np.in1d( vID_Trig_all, np.array([ 3049419,  9020431,  9172305,  9173365, 15332137]))
    # print( "события в trig_all перед удалением дубликатов: ", sel_same.sum(), vID_Trig_all[sel_same])
    aIDs, aCounts = np.unique( vID_Trig_all, return_counts=True)
    selDouble = aCounts > 1
    if verbose == True:
        print( f"Количество ID событий в нескольких кластерах: {selDouble.sum()}")
    i_run = 1
    while selDouble.sum() > 0:
        if verbose == True:
            print( '%i. прогон для удаления дубликатов'%(i_run))
        for ID in np.unique( aIDs[selDouble]):
            selCl = ID == vID_Trig_all
            aClID = np.unique( vclID_allEv[selCl])
            for iCl in range( len( aClID)-1):
                if verbose == True:
                    print( 'Кластеры с одинаковыми событиями', str( aClID[0]), str( aClID[iCl+1]), 'evID: ', int(ID))
                #A# объединяем кластеры с одинаковыми событиями
                dClust[str(aClID[0])] = np.unique( np.hstack( (dClust[str(int( aClID[0]))], dClust[str( int(aClID[iCl+1]))])))
                #B# удаляем ID кластеров из словаря
                dClust.pop( str( int( aClID[iCl+1])))
            #C# удаляем дублирующее событие и соответствующий clID из:
            # vID_Trig_all
            sel_rem = ID != vID_Trig_all
            vID_Trig_all = vID_Trig_all[sel_rem]
            # и vclID_allEv
            vclID_allEv  = vclID_allEv[sel_rem]
            # оставляем одно событие с новым clID, т.е. clID первого кластера, содержащего ID
            vclID_allEv  = np.append( vclID_allEv, aClID[0])
            vID_Trig_all = np.append( vID_Trig_all, ID)
        aIDs, aCounts = np.unique( vID_Trig_all, return_counts=True)
        selDouble = aCounts > 1
        i_run += 1
    # находим события в начальной выборке одиночных (eta > eta_0),
    # которые на самом деле являются частью кластеризованных событий
    sel_single = np.ones( vID_single.shape[0], dtype = int) > 0
    iS = 0
    for ID_single in  vID_single:
        sel = ID_single == vID_Trig_all
        if sel.sum() > 0: # удаляем это событие из одиночных
            sel_single[iS] = False
        iS += 1
    if verbose == True:
        print("Начальные одиночные события, ставшие родительскими - удаляем из dClust['0']: ",np.array([~sel_single]).sum())

    vID_single = vID_single[sel_single]
    if verbose == True:
        print("---------------Итоговый результат--------------------------")
        print(f"Общее количество в кластерах: {len(vID_Trig_all)}"), 
        print(f"Количество родителей (=количество кластеров): {len(dClust.keys())},"),
        print(f"Количество одиночных: {vID_single.shape[0]}, Общее количество потомков (включая дубликаты):  {nTotChild}"),
        print("Доля триггерных: ", round((len( vID_Trig_all)-len(dClust.keys()))/dNND['aNND'].shape[0],2), "Доля главных толчков: ", round( len(dClust.keys())/dNND['aNND'].shape[0],2), "Одиночные: ", round((vID_single.shape[0]/dNND['aNND'].shape[0]),2))
        print('Общее количество в каталоге: ', dNND['aNND'].shape[0]+1), 
        print('Триггерные + одиночные', len(vID_Trig_all) + vID_single.shape[0])

    dClust[str(0)] =  vID_single
    return dClust

def addClID2cat( seisCat, dClust, test_plot = False, **kwargs):
    """
    - Добавление нового столбца (т.е. тега словаря='famID') для seisCat,
      который указывает, к какому кластеру относится каждое событие
    - !Обратите внимание, что если нужно записать поколение потомков, сначала выполните:
      clustering.offspring_gen() и используйте выходной словарь как вход для этой функции

    :param dClust:  словарь Python
                    каждый элемент словаря, заданный ключом, — вектор ID событий
                    или матрица из трёх строк с evID, iGen и средней глубиной листа


    :param seisCat:
    :return: seisCat (с новыми тегами:
                        'famID' - запись семейных связей между событиями
                        опционально:
                        (обратите внимание, что 'clID' обычно используется для релокации на основе волновых форм)
                        'iGen'  - запись поколения потомков в семье
                        'LD'    - средняя глубина листа для каждого кластера
    """
    # сортируем исходный каталог, чтобы получить ID первого события
    seisCat.sortCatalog( 'Time')

    # первая строка — ID кластера, вторая строка — ID события из каталога
    nRows  = 2
    b_add_iGen = False
    if len( dClust['0'].shape) > 1:
        b_add_iGen = True
        # дополнительные строки для поколения триггера и средней глубины листа
        nRows = 4
    mClust = np.zeros([nRows, seisCat.size()])
    nGen = 0
    nFam = 0
    nEv = 0
    i   = 0
    for sCl in dClust.keys():
        iCl = int(sCl)
        # print( f"------iCl: {iCl}, nEv: {nEv}--------, evID={dClust[sCl]}")
        # ID событий землетрясений
        if b_add_iGen == False:
            nEv = dClust[sCl].shape[0]
            mClust[1, i:i + nEv] = dClust[sCl]
        else:
            nEv = dClust[sCl].shape[1]
            mClust[1, i:i + nEv] = dClust[sCl][0]
        # ID семей
        mClust[0,i:i+nEv] = np.ones(nEv)*iCl
        nFam += len( dClust[sCl])
        if b_add_iGen == True:
            nGen += len( dClust[sCl][1])
            # поколение триггера
            mClust[2,i:i+nEv] = dClust[sCl][1]
            # средняя глубина листа
            mClust[3, i:i + nEv] = dClust[sCl][2]
        i += nEv
    #---------включаем первое событие в каталоге как одиночное-------------
    selFirst = seisCat.data['N'][0] == mClust[1]
    if selFirst.sum() == 0:
        ID_first = int( seisCat.data['N'][0]) #[~selUni][0])
        print( 'Первое событие в каталоге - ID:', ID_first, int( seisCat.data['N'][0]), 'Последнее событие в mClust', mClust[1,-1], 'должно быть = 0')
        mClust[1] = np.hstack( (ID_first, mClust[1,0:-1]))# не требуется, если каталог отсортирован по времени
    #sel_same = np.in1d( mClust[1], seisCat.data['N'])
    # проверяем, что каждый ID события представлен только один раз
    __, aID, aN_uni = np.unique( mClust[1], return_counts = True, return_index=True)
    sel = aN_uni > 1
    if sel.sum() > 0:
        error_str = f"ID события представлен более одного раза: {mClust[1][aID[sel]]}, 'Количество повторов: ', {aN_uni[sel]}"
        raise ValueError( error_str)
    #--сортируем матрицу ID кластеров и каталог относительно ID
    sortSel = mClust[1].argsort()
    mClust = mClust.T[sortSel].T
    seisCat.sortCatalog('N') #--иначе clID будут назначены неверным событиям

    if test_plot == True:
        plt.figure()
        plt.subplot( 211)
        plt.plot( mClust[1], mClust[1]-seisCat.data['N'], 'ko')
        plt.xlabel( 'ID события в кластере')
        plt.ylabel( 'Разница ID событий (0)')
        plt.subplot( 212)
        plt.plot(mClust[1],  mClust[0], 'ko')
        plt.xlabel('ID события в кластере')
        plt.ylabel('ID кластера')
        #plt.plot( plt.gca().get_xlim(), plt.gca().get_xlim(), 'r--')
        plt.show()

    seisCat.data['famID'] = np.int32( mClust[0])
    if b_add_iGen == True:
        seisCat.data['iGen'] = np.int16(mClust[2])
    seisCat.sortCatalog( 'Time')
    return seisCat

def offspring_gen( dClust, dNND, f_eta_0, **kwargs):
    """
    - Отслеживание цепочки триггеров хронологически и присвоение поколения триггера
    - Начинаем с родительского поколения, затем добавляем конечные листья
            a) определяем всех родителей в кластере
            b) сортируем по времени
            c) присваиваем одинаковое iGen потомкам одного родителя (иерархически)
            Вычисляем среднюю глубину листа:
                <d> = 1/n sum( d_i) = средняя глубина по конечным листам
    __________________________________
    Входные данные:  seisCat   = объект SeismicityCatalog
                        используется для получения времени возникновения дочерних событий
            dNND =
            'aEqID_c'  - уникальные ID дочерних событий
            'aEqID_p ' - ID родительских событий, пары с a_ID_child, порядок важен
                          родители могут иметь много потомков, поэтому повторы возможны
            'Time'     - время возникновения дочерних событий из каталога, если ID не хронологические

            dClust - '[famID]' = np.array([ offSpringIDs])
    ----------------------------------
    Возвращает:
            dGen    - словарь Python
                    'famID' : np.array([3, N])
                    # dGen[famID][0] = evIDs
                    # dGen[famID][1] = поколение триггера
                    # dGen[famID][2] = средняя глубина листа
                             - средняя глубина листа (одинаковое число для всего кластера)
    """
    #=========================1========================================
    #            подсчёт поколений дочерних событий
    #==================================================================
    dGen = {}
    l_famID = list( dClust.keys())
    # одиночные события — все 0 поколения
    dGen['0'] = np.zeros( (3, len( dClust['0'])))
    # устанавливаем ID событий в новом словаре
    dGen['0'][0] = dClust['0']

    # средняя глубина листа = 1
    dGen['0'][2] = np.ones( len( dClust['0']))

    # игнорируем одиночные события ниже
    l_famID.remove( '0')
    for famID in l_famID:
        ###находим время возникновения для каждого дочернего события
        sel_chi_t  = np.in1d(  dNND['aEqID_c'], dClust[famID])
        # фильтруем для пар родитель-потомок с NND < eta_0
        sel_chi_t2 = dNND['aNND'][sel_chi_t] < f_eta_0
        curr_iPar  = dNND['aEqID_p'][sel_chi_t][sel_chi_t2]
        curr_iChi  = dNND['aEqID_c'][sel_chi_t][sel_chi_t2]
        curr_tChi  = dNND['Time'][np.in1d( dNND['aEqID_c'],curr_iChi)]

        ##сортируем ID кластеров относительно времени потомков
        sel_sort  = np.argsort( curr_tChi)
        first_ID  = curr_iChi[sel_sort][0]

        # получаем уникальных родителей и сортируем по времени
        uni_curr_iPar = np.unique(curr_iPar)
        uni_par_times = curr_tChi[np.in1d(curr_iChi, uni_curr_iPar)]
        uni_curr_iPar = curr_iChi[np.in1d(curr_iChi, uni_curr_iPar)]
        sort_uni_par  = np.argsort( uni_par_times)
        uni_curr_iPar = uni_curr_iPar[sort_uni_par]
        # проверяем, нужно ли добавить родителя первой пары
        if np.isin( curr_iPar[0], uni_curr_iPar).sum() == 0:
            uni_curr_iPar = np.hstack(( curr_iPar[0], uni_curr_iPar))
        # добавляем конечные листья (потомков, которые не являются родителями)
        sel_endLeaf   = ~np.in1d(curr_iChi, curr_iPar)
        uni_curr_iPar = np.hstack((uni_curr_iPar, curr_iChi[sel_endLeaf]))
        #----------инициализируем новые векторы------------------------
        uni_iGen_pastPar = np.zeros( len(curr_tChi)+1)
        uni_id_pastPar   = np.zeros( len(curr_tChi)+1)
        ## присваиваем хронологическое поколение триггера
        curr_iGen        = np.zeros( len(curr_tChi)+1)
        iGen          = 0
        for iPar in range( len(uni_curr_iPar)):
            # проверяем, является ли текущий родитель потомком другого родителя
            pastPar = curr_iPar[uni_curr_iPar[iPar] == curr_iChi]
            if len( pastPar) > 0:
                sel_pastPar = pastPar == uni_id_pastPar
            else:
                sel_pastPar = np.array([False])
            if sel_pastPar.sum() > 0:
                # добавляем 1 к поколению предыдущего родителя
                curr_iGen[iPar]         = uni_iGen_pastPar[sel_pastPar][0]+1
                uni_iGen_pastPar[iPar]  = uni_iGen_pastPar[sel_pastPar][0]+1
                uni_id_pastPar[iPar]    = uni_curr_iPar[iPar]
            else:
                curr_iGen[iPar]         = iGen
                uni_iGen_pastPar[iPar]  = iGen
                uni_id_pastPar[iPar]    = uni_curr_iPar[iPar]
                iGen += 1  # присваиваем новое поколение триггера
        # сохраняем ID события, поколение триггера в словаре
        dGen[famID]    = np.zeros( (3, len(uni_id_pastPar)))

        dGen[famID][0] = uni_id_pastPar
        dGen[famID][1] = curr_iGen
        # =========================3========================================
        #             вычисляем среднюю глубину листа
        # ==================================================================
        sel_endLeaf = ~np.in1d(curr_iChi, curr_iPar)
        dGen[famID][2] = np.ones( len(dClust[famID]))*curr_iGen[1::][sel_endLeaf].mean()
    return dGen

def offspring_gen_test( dClust, dNND, f_eta_0, **kwargs):
    #=========================1========================================
    #               добавляем время возникновения из seisCat в dNND
    #==================================================================
    # сортируем dNND и seisCat по ID потомков!!- seisCat.data['Time'] добавляется в dNND
    # sortSel = np.argsort( dNND['aEqID_c'])
    # for tag in list(dNND.keys()):
    #     dNND[tag] = dNND[tag][sortSel]
    # seisCat.sortCatalog('Time')
    # firstEvID = seisCat.data['N'][0]
    # seisCat.sortCatalog('N')
    # # добавляем время возникновения потомков в dNND
    # sel = firstEvID == seisCat.data['N']
    # dNND['at_c'] = seisCat.data['Time'][~sel]
    # проверяем, что dNND отсортирован по времени
    # if 'Time' in dNND.keys():
    #     i_sort = np.argsort( dNND['Time'])
    #     dNND   = data_utils.selDicAll(dNND, i_sort)
    # else:
    #     error_str = "Отсутствует ключ 'Time', добавьте время возникновения дочерних событий в dNND"
    #     raise ValueError( error_str)
    #=========================2========================================
    #            подсчёт поколений дочерних событий
    #==================================================================
    dGen = {}
    l_famID = list( dClust.keys())
    # одиночные события — все 0 поколения
    dGen['0'] = np.zeros( (3, len( dClust['0'])))
    # устанавливаем ID событий в новом словаре
    dGen['0'][0] = dClust['0']
    # средняя глубина листа = 1
    dGen['0'][2] = np.ones( len( dClust['0']))
    # игнорируем одиночные события ниже
    l_famID.remove( '0')
    for famID in l_famID:
        ###находим время возникновения для каждого дочернего события
        sel_chi_t  = np.in1d(  dNND['aEqID_c'], dClust[famID])
        # фильтруем для пар родитель-потомок с NND < eta_0
        sel_chi_t2 = dNND['aNND'][sel_chi_t] < f_eta_0
        curr_iPar  = dNND['aEqID_p'][sel_chi_t][sel_chi_t2]
        curr_iChi  = dNND['aEqID_c'][sel_chi_t][sel_chi_t2]
        curr_tChi  = dNND['Time'][np.in1d( dNND['aEqID_c'],curr_iChi)]
        # curr_tChi = np.zeros( len( curr_iPar))
        # for iP in range( len( curr_iChi)):
        #     curr_tChi[iP] = dNND['at_c'][dNND['aEqID_c']==curr_iChi[iP]]

        ##сортируем ID кластеров относительно времени потомков
        sel_sort  = np.argsort( curr_tChi)

        curr_tChi = curr_tChi[sel_sort]
        curr_iChi = curr_iChi[sel_sort]
        curr_iPar = curr_iPar[sel_sort]

        # ID родителей имеют на один элемент меньше, чем полный кластер (первое событие не имеет родителя)
        #sel_sort =  np.hstack((0, sel_sort+1))
        # убеждаемся, что dClust[famID] = dNND['aEqID_c']+
        firstID = dClust[famID][0]

        uni_curr_iPar = np.unique( curr_iPar)
        # сортируем уникальных родителей по времени
        print( uni_curr_iPar)
        uni_par_times = curr_tChi[np.in1d(curr_iChi, uni_curr_iPar)]
        uni_curr_iPar = curr_iChi[np.in1d(curr_iChi, uni_curr_iPar)]
        sort_uni_par  = np.argsort( uni_par_times)
        uni_curr_iPar = uni_curr_iPar[sort_uni_par]
        # проверяем, нужно ли добавить родителя первой пары
        if np.isin( curr_iPar[0], uni_curr_iPar).sum() == 0:
            uni_curr_iPar = np.hstack(( curr_iPar[0], uni_curr_iPar))
        # добавляем конечные листья (потомков, которые не являются родителями)
        sel_endLeaf   = ~np.in1d(curr_iChi, curr_iPar)
        uni_curr_iPar = np.hstack((uni_curr_iPar, curr_iChi[sel_endLeaf]))
        #----------инициализируем новые векторы------------------------
        uni_iGen_pastPar = np.zeros( len(curr_tChi)+1)
        uni_id_pastPar   = np.zeros( len(curr_tChi)+1)
        ## присваиваем хронологическое поколение триггера
        curr_iGen        = np.zeros( len(curr_tChi)+1)
        iGen          = 0
        for iPar in range( len(uni_curr_iPar)):
            # # присваиваем поколение триггера, начиная с самого старого родителя
            # sel_hier_par = curr_iPar == uni_curr_iPar[iPar]
            # # print("текущий родитель: ", uni_curr_iPar[iPar], "потомки: ", curr_iChi[sel_hier_par])
            # # print( "прошлые родители", uni_id_pastPar)
            # # print( "поколение прошлых родителей", uni_iGen_pastPar)
            # проверяем, является ли текущий родитель потомком другого родителя
            pastPar = curr_iPar[uni_curr_iPar[iPar] == curr_iChi]
            #print( uni_curr_iPar[iPar], pastPar, uni_id_pastPar)
            if len( pastPar) > 0:
                sel_pastPar = pastPar == uni_id_pastPar
            else:
                sel_pastPar = np.array([False])
            if sel_pastPar.sum() > 0:
                print(uni_curr_iPar[iPar], "прошлый родитель: ", pastPar, "поколение триггера: ", uni_iGen_pastPar[sel_pastPar][0]+1)
                # добавляем 1 к поколению предыдущего родителя
                curr_iGen[iPar]         = uni_iGen_pastPar[sel_pastPar][0]+1
                uni_iGen_pastPar[iPar]  = uni_iGen_pastPar[sel_pastPar][0]+1
                uni_id_pastPar[iPar]    = uni_curr_iPar[iPar]
            else:
                print( "поколение триггера: ", iGen)
                curr_iGen[iPar]         = iGen
                uni_iGen_pastPar[iPar]  = iGen
                uni_id_pastPar[iPar]    = uni_curr_iPar[iPar]
                iGen += 1  # присваиваем новое поколение триггера

        print( uni_id_pastPar)
        print( curr_iGen)
        # сохраняем ID события, поколение триггера в словаре
        dGen[famID]    = np.zeros( (3, len(uni_id_pastPar)))
        dGen[famID][0] = uni_id_pastPar
        dGen[famID][1] = curr_iGen
        # =========================3========================================
        #             вычисляем среднюю глубину листа
        # ==================================================================
        # конечные листья = события без потомков, curr_iPar уже < eta_0)
        #print( len( curr_iGen), len( curr_iChi), len( uni_curr_iPar))
        sel_endLeaf = ~np.in1d(curr_iChi, curr_iPar)
        print( 'ID конечных потомков', curr_iChi[sel_endLeaf])
        print( 'глубины листа ', curr_iGen[1::][sel_endLeaf])
        print( 'средняя глубина листа: ', curr_iGen[1::][sel_endLeaf].mean())
        dGen[famID][2] = np.ones( len(dClust[famID]))*curr_iGen[1::][sel_endLeaf].mean()
    # структура данных:
    # dGen[famID][0] = ID событий
    # dGen[famID][1] = поколение триггера
    # dGen[famID][2] = средняя глубина листа
    return dGen
#=================================================================================
#                      создание случайных каталогов
#=================================================================================
# создание равномерных времён
def rand_rate_uni( N, tmin, tmax, **kwargs):
    """  Выборка N случайных чисел из пуассоновского распределения, заданного mu, между tmin и tmax,

    kwargs: - случайная равномерная переменная между min и max

    Возвращает: вектор из N времён возникновения между tmin и tmax """
    return np.random.uniform( tmin, tmax, size = N)


# ------------------------------------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2, **kwargs):
    """
    Реализация формулы гаверсинуса
    https://en.wikipedia.org/wiki/Great-circle_distance
    Расстояние по большому кругу между двумя точками
    :Входные данные   lon1, lat1
             lon2, lat2

    		  gR - радиус Земли (глобальная переменная)
    :Выходные данные  расстояние - расстояние по большому кругу в километрах
    """
    i_radius = 6371
    # конвертируем в радианы
    lon1 = lon1 * np.pi / 180
    lon2 = lon2 * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    # формула гаверсинуса
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = i_radius * c
    return distance

# ==================================4==============================================================
#                       графики плотности T-R
# =================================================================================================
def plot_R_T( a_T, a_R, f_eta_0, **kwargs):
    """
        - Построение графика масштабированного расстояния против масштабированного времени
        Параметры:
    dPar = {'binx': .1, 'biny': .1,  # используется для плотности и гауссова сглаживания
            'sigma': None,  # если None: по умолчанию = n**(-1./(d+4)),
            'Tmin': -8, 'Tmax': 0,
            'Rmin': -5, 'Rmax': 3,
            'cmap': plt.cm.RdYlGn_r}
    Используйте kwargs['dPar'] = словарь Python
                        'binx', 'biny' и т.д. для переписывания
                        значений по умолчанию для конкретных или всех параметров
    :param kwargs:
    :return: fig - дескриптор фигуры - используйте fig.axes для получения списка соответствующих осей
    """
    dPar = {'binx': .1, 'biny': .1,  # используется для плотности и гауссова сглаживания
            'sigma': None,  # если None: по умолчанию = n**(-1./(d+4)),
            'Tmin': -8, 'Tmax': 0,
            'Rmin': -5, 'Rmax': 3,
            'cmap': plt.cm.RdYlGn_r}
    if 'dPar' in kwargs.keys() and kwargs['dPar'] is not None:
        for tag in kwargs['dPar'].keys():
            print( f"Переписываем параметр plot_R_T: {tag}={kwargs['dPar'][tag]}")
            dPar[tag] = kwargs['dPar'][tag]
    a_Tbin = np.arange(dPar['Tmin'], dPar['Tmax'] + 2 * dPar['binx'], dPar['binx'])
    a_Rbin = np.arange(dPar['Rmin'], dPar['Rmax'] + 2 * dPar['biny'], dPar['biny'])
    sel = a_T > 0
    XX, YY, ZZ = data_utils.density_2D(np.log10(a_T[sel]), np.log10(a_R[sel]), a_Tbin, a_Rbin, sigma=dPar['sigma'])

    fig = plt.figure( figsize=(7, 9))
    ax = plt.subplot(111)
    ax.set_title('Пары ближайших соседей в R-T')
    # ------------------------------------------------------------------------------
    normZZ = ZZ * (dPar['binx'] * dPar['biny'] * len(a_R))
    plot1 = ax.pcolormesh(XX, YY, normZZ, cmap=dPar['cmap'])
    cbar = plt.colorbar(plot1, orientation='horizontal', shrink=.5, aspect=20, )
    # ax.plot(  np.log10( a_T), np.log10( a_R), 'wo', ms = 1.5, alpha = .2)
    # построение eta_0 для разделения кластерного и фонового режимов
    ax.plot([dPar['Tmin'], dPar['Tmax']], -np.array([dPar['Tmin'], dPar['Tmax']]) + f_eta_0, '-', lw=1.5, color='w')
    ax.plot([dPar['Tmin'], dPar['Tmax']], -np.array([dPar['Tmin'], dPar['Tmax']]) + f_eta_0, '--', lw=1.5, color='.5')
    # -----------------------надписи и легенды-------------------------------------------------------
    # cbar.set_label( 'Плотность пар событий [#событий/dRdT]')
    cbar.set_label('Количество пар событий', labelpad=-60)
    ax.set_xlabel('Масштабированное время')
    ax.set_ylabel('Масштабированное расстояние')
    ax.set_xlim(dPar['Tmin'], dPar['Tmax'])
    ax.set_ylim(dPar['Rmin'], dPar['Rmax'])
    # fig.axes
    return fig