

def sum(dic1, dic2):
    if type(dic1) is dict:
        return {k: sum(dic1[k], dic2[k]) for k in dic1}

    return dic1+dic2


def multiply(dic1, value):
    if type(dic1) is dict:
        return {k: multiply(dic1[k], value) for k in dic1}

    return dic1*value


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
            assert False, f' {x} is in dic1={dic1}  but not in dic2 {dic2}'
        if not all(k in dic1 for k in dic2):
            x = [k for k in dic2 if k not in dic1]
            assert False, f' {x} is in dic2={dic2}  but not in dic1 {dic1}'

        for k in dic2:
            assert_same_keys(dic1[k], dic2[k])

    return True
