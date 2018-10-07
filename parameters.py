# Module containing dictionary of parameters

param = {'m1000001' : None,
         'm1000002' : None,
         'm1000003' : None,
         'm1000004' : None,
         'm1000005' : None,
         'm1000006' : None,
         'm2000001' : None,
         'm2000002' : None,
         'm2000003' : None,
         'm2000004' : None,
         'm2000005' : None,
         'm2000006' : None,
         'm1000021' : None}


def import_slha():
    print 'Not implemented!'

def set_parameter(name, value):
    if name not in param.keys():
        print 'Paramter name %s not known!' % name
        raise KeyError
    param[name] = value

def set_parameters(params_in):
    for nam,val in params_in.items():
        set_parameter(nam, val)

#set_parameter('rar',700)
#set_parameters({'m1000001':700,'m1000001':600})
#print param['m1000001'], param['m1000002']
