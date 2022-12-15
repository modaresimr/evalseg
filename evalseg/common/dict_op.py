

def sum(dic1, dic2):
    if type(dic1) is dict:
        return {k: sum(dic1[k], dic2[k]) for k in dic1}

    return dic1+dic2


def concat(dic1, dic2):
    if type(dic1) is dict:
        return {k: concat(dic1[k], dic2[k]) for k in dic1}
    if type(dic1) != tuple:
        dic1 = (dic1,)
    if type(dic2) != tuple:
        dic2 = (dic2,)

    return dic1+dic2


def multiply(dic1, value):
    if type(dic1) is dict:
        return {k: multiply(dic1[k], value) for k in dic1}

    return dic1*value


def apply_func(dic1, func):
    if type(dic1) is dict:
        return {k: apply_func(dic1[k], func) for k in dic1}

    return func(dic1)


def have_same_keys(dic1, dic2):
    if type(dic1) != dict and type(dic2) == dict:
        return False
    if type(dic1) == dict and type(dic2) != dict:
        return False

    if type(dic1) == dict and type(dic2) == dict:
        if not all(k in dic2 for k in dic1):
            return False
        if not all(k in dic1 for k in dic2):
            return False

        for k in dic2:
            if not have_same_keys(dic1[k], dic2[k]):
                return False

    return True


def assert_same_keys(dic1, dic2):
    if type(dic1) != dict and type(dic2) == dict:
        assert False, f'type {dic1} is different from type {dic2}'
    if type(dic1) == dict and type(dic2) != dict:
        assert False, f'type {dic1} is different from type {dic2}'

    if type(dic1) == dict and type(dic2) == dict:
        if not all(k in dic2 for k in dic1):
            x = [k for k in dic1 if k not in dic2]
            assert False, f' {x} is in dic1={dic1.keys()}  but not in dic2 {dic2.keys()}'
        if not all(k in dic1 for k in dic2):
            x = [k for k in dic2 if k not in dic1]
            assert False, f' {x} is in dic2={dic2.keys()}  but not in dic1 {dic1.keys()}'

        for k in dic2:
            assert_same_keys(dic1[k], dic2[k])

    return True


def common_keys(dic1, dic2):
    if type(dic1) != dict and type(dic2) == dict:
        print(f'Warning! type {dic1} is different from type {dic2}')
        return None
    if type(dic1) == dict and type(dic2) != dict:
        print(f'Warning! type {dic1} is different from type {dic2}')
        return None

    if type(dic1) == dict and type(dic2) == dict:

        if not all(k in dic2 for k in dic1):
            x = [k for k in dic1 if k not in dic2]
            print(f'warning! {x} is in dic1 but not in dic2')
        if not all(k in dic1 for k in dic2):
            x = [k for k in dic2 if k not in dic1]
            print(f'warning! {x} is in dic2 but not in dic1')

        com_keys = [k for k in dic1 if k in dic2]
        ret = {k: common_keys(dic1[k], dic2[k]) for k in com_keys}
        return {k: ret[k][0] for k in com_keys if ret[k] is not None}, {k: ret[k][1] for k in com_keys if ret[k] is not None}

    return dic1, dic2
