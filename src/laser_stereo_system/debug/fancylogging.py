from constants import Logging
codes = Logging.PrintCodes

def print_ansi(msg: str):
    '''Prints a message containig ANSI escape codes with append clear code at the end.'''
    print(f"{msg}{codes.ENDC}")

def log_header(msg: str):
    if Logging.LOG_LEVEL == Logging.LogLevels.INFO:
        print_ansi(f"\n{codes.BOLD}{codes.UNDERLINE}{codes.HEADER}{msg}")

def log_ok(msg: str):
    '''Prints a message indicating OK status from program'''
    if Logging.LOG_LEVEL == Logging.LogLevels.INFO:
        print_ansi(f"{codes.OKCYAN}{codes.UNDERLINE}OK{codes.NOUNDERLINE}: {msg}")

def log_info(msg: str):
    '''Prints an info message from program'''
    if Logging.LOG_LEVEL == Logging.LogLevels.INFO:
        print_ansi(f"{codes.INFOWHITE}{codes.UNDERLINE}INFO{codes.NOUNDERLINE}: {msg}")

def log_warn(msg: str):
    '''Prints a message indicating a warning from the program'''
    if Logging.LOG_LEVEL >= Logging.LogLevels.WARN:
        print_ansi(f"{codes.WARNING}{codes.UNDERLINE}WARN{codes.NOUNDERLINE}: {msg}")

def log_err(msg: str, throw=False, err=Exception()):
    '''Prints a message indicating an error in the program, optionally raising an error'''
    if Logging.LOG_LEVEL >= Logging.LogLevels.ERR:
        print_ansi(f"{codes.FAIL}{codes.UNDERLINE}ERROR{codes.NOUNDERLINE}: {msg}")
    if throw: raise err