'''
methods for preprocessing 
'''
def position2annotation(p):
    panel_scale = 50
    panel_width = 1920
    panel_height = 1080
    px = p[0]
    py = p[1]
    ax = (px * panel_scale / panel_width + 0.5) * panel_width
    ay = (py * panel_scale / panel_height + 0.5) * panel_height
    return ax, ay

def annotation2position(a):
    panel_scale = 50
    panel_width = 1920
    panel_height = 1080
    ax = a[0]
    ay = a[1]
    px = (ax / panel_width - 0.5 ) * panel_width/ panel_scale
    py = (ay / panel_height - 0.5 ) * panel_height/ panel_scale
    return px, py

def count_total_human(human_ids):
    human_set = set(human_ids[0])
    for h_ids in human_ids:
        human_set = human_set | set(h_ids)

def count_total_human_from_f(f, start_t, end_t):
    p_set = set(f[start_t])
    for t in range(start_t, end_t +1):
        p_set = p_set | set(f[t])
    return p_set

def if_f_p_accord(f_data_list, p_data_list):
    newf = [[] for i in range(7000)]
    for i in range(len(p_data_list)):
        p = p_data_list[i]
        tlist = [int(t) for t in list(p.keys())]
        for t in tlist:
            newf[t].append(i)
    print(f_data_list == newf[:5000])

def make_new_f(p_data_list):
    newf = [[] for i in range(7000)]
    for i in range(len(p_data_list)):
        p = p_data_list[i]
        tlist = [int(t) for t in list(p.keys())]
        for t in tlist:
            newf[t].append(i)
    return newf


def check_continue(a):
    b = [i for i in range(min(a), min(a) + len(a))]
    return a == b


def add_positions(tra):
    newtra = {}
    tlist = [int(t) for t in list(tra.keys())]
    newtlist = [i for i in range(min(tlist), max(tlist) + 1)]
    for t in newtlist:
        if t in tlist:
            newtra[str(t)] = tra[str(t)]
        else:
            newtra[str(t)] = newtra[str(t - 1)]
    return newtra