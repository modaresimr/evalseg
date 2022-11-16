

def sum(dic1, dic2):
    if type(dic1) is dict:
        return {k: sum(dic1[k], dic2[k]) for k in dic1}

    return dic1+dic2


def multiply(dic1, value):
    if type(dic1) is dict:
        return {k: multiply(dic1[k], value) for k in dic1}

    return dic1*value


def have_same_keys(dic1, dic2):
    if type(dic1) != type(dic2):
        return False

    if type(dic1) is dict:
        if not all(k in dic2 for k in dic1):
            return False

        for k in dic2:
            if not have_same_keys(dic1[k], dic2[k]):
                return False

    return True
