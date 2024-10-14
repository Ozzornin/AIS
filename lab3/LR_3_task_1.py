def xor(x1, x2):
    or_result = or_(x1, x2)
    and_result = and_(x1, x2)
    return or_result and not and_result


def or_(x1, x2):
    return x1 or x2


def and_(x1, x2):
    return x1 and x2


# Тестування
print(xor(False, False))  # Очікується: False (0)
print(xor(False, True))  # Очікується: True (1)
print(xor(True, False))  # Очікується: True (1)
print(xor(True, True))  # Очікується: False (0)
