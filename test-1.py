
born = '08/16/1999'
today = '08/26/2019'

def calculate_age(born, today):
    age, d, m, y = 0, 0, 0, 0
    bm, bd, by = int(born.split('/')[0]), int(born.split('/')[1]), int(born.split('/')[2])
    tm, td, ty = int(today.split('/')[0]), int(today.split('/')[1]), int(today.split('/')[2])
    if td >= bd: d = td-bd
    else:
        d = (td+30) - bd
        tm = tm-1
    if tm >=bm: m =tm-bm
    else:
        m = (tm+12) - bm
        ty = ty-1
    y = ty-by
    age = round((d+(m*30)+(y*365))/365, 2)
    print(age)
calculate_age(born, today)


def snapToHandoff(TimeHandoff, TimeSnap):
    return int(TimeHandoff.split('.')[0].split(':')[-1]) - int(TimeSnap.split('.')[0].split(':')[-1])

TimeSnap = '2017-09-08T00:44:05.000Z'
TimeHandoff = '2017-09-08T00:44:06.000Z'
print(snapToHandoff(TimeHandoff, TimeSnap))
