from numba import jit

class Logging:
    '''Class for logging to console with various logging levels.'''
    class LogLevels:
        ERR = 1
        WARN = 2
        INFO = 3
    LOG_LEVEL = LogLevels.INFO

    class PrintCodes:
        HEADER = '\033[95m'
        INFOWHITE = '\033[97m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        NOUNDERLINE = '\033[24m'

codes = Logging.PrintCodes
LOG_LEVEL = Logging.LOG_LEVEL
LogLevels = Logging.LogLevels

@jit
def print_ansi(msg: str):
    '''Prints a message containig ANSI escape codes with append clear code at the end.'''
    print(f"{msg}{codes.ENDC}")

@jit
def log_header(msg: str):
    if LOG_LEVEL == LogLevels.INFO:
        print_ansi(f"\n{codes.BOLD}{codes.UNDERLINE}{codes.HEADER}{msg}")

@jit
def log_ok(msg: str):
    '''Prints a message indicating OK status from program'''
    if LOG_LEVEL == LogLevels.INFO:
        print_ansi(f"{codes.OKCYAN}{codes.UNDERLINE}OK{codes.NOUNDERLINE}: {msg}")

@jit
def log_info(msg: str):
    '''Prints an info message from program'''
    if LOG_LEVEL == LogLevels.INFO:
        print_ansi(f"{codes.INFOWHITE}{codes.UNDERLINE}INFO{codes.NOUNDERLINE}: {msg}")

@jit
def log_warn(msg: str):
    '''Prints a message indicating a warning from the program'''
    if LOG_LEVEL >= LogLevels.WARN:
        print_ansi(f"{codes.WARNING}{codes.UNDERLINE}WARN{codes.NOUNDERLINE}: {msg}")

@jit
def log_err(msg: str, throw=False, err=Exception()):
    '''Prints a message indicating an error in the program, optionally raising an error'''
    if LOG_LEVEL >= LogLevels.ERR:
        print_ansi(f"{codes.FAIL}{codes.UNDERLINE}ERROR{codes.NOUNDERLINE}: {msg}")
    if throw: raise err