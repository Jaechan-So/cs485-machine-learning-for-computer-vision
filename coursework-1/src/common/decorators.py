import time

tag_parameter_name = 'tag'


def measure_time(*args_decorator, **kwargs_decorator):
    if tag_parameter_name in kwargs_decorator and kwargs_decorator[
        tag_parameter_name] is not None:
        tag = f"{kwargs_decorator[tag_parameter_name]}"
    else:
        tag = ''

    def wrapper(fn):
        def inner(*args_inner, **kwargs_inner):
            start_time = time.time()
            result = fn(*args_inner, **kwargs_inner)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            print(f'[{tag}] Elapsed time: {elapsed_time_ms:.2f} milliseconds')

            return result

        return inner

    return wrapper
