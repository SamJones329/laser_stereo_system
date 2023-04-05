try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time
class PerfTimer:
    '''Timer for getting runtime performance statistics'''
    
    num_runs = 0
    total_runtime = 0.
    debug = False
    
    def __init__(self, debug=False):
        '''Makes a new performance timer'''
        self.debug = debug
        self._self_str = "<PerfTimer@%s>" % hex(id(self))
    
    def start(self):
        '''Starts the timer'''
        self.start_time = perf_counter()
        if self.debug: print("%s: Started timer at %.04f" % (self._self_str, self.start_time))
    
    def stop(self):
        '''Stops the timer and records the run'''
        self.stop_time = perf_counter()
        self.num_runs += 1
        self.total_runtime += self.stop_time - self.start_time
        if self.debug: 
            print("%s: Stopped timer at %.04fs" % (self._self_str, self.stop_time))
            print("%s: Timer ran for %.04fs" % (self._self_str, self.get_last_runtime))

    def get_last_runtime(self):
        '''Gets the time in seconds of the last stop call minus the time of the last start call.'''
        return self.stop_time - self.start_time
    
    def get_avg_runtime(self):
        '''Gets the average runtime of the timer in seconds'''
        try:
            return self.total_runtime / self.num_runs
        except: 
            return 0.
    
    def __str__(self):
        return "%s{ Runs: %d, Total Runtime: %.04fs, Average Runtime: %.04fs }" % (self._self_str, self.num_runs, self.total_runtime, self.get_avg_runtime())