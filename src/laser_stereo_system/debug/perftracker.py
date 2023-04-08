import time
import tracemalloc
from functools import wraps
from typing import Callable
import pandas as pd
import numpy as np

DEBUG_ENABLED = False

class PerfTracker:
    '''Tracker for runtime performance statistics'''

    # create process tracker with id and name, return id
    # create start tracking and stop tracking methods that start and stop timer for given id
    # keep expanding array of time samples for each process being tracked

    tracking_data = {}

    def track(name: str):
        if name in PerfTracker.tracking_data:
            raise ValueError(f"Already tracking a function called {name}")
        PerfTracker.tracking_data[f"{name}"] = {
            "runtimes": [],
            "maxmems": []
        }
        def track_decorator(func: Callable):
            @wraps(func)
            def perftrack_wrapper(*args, **kwargs):
                tracemalloc.start()
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                mem_trace = tracemalloc.get_traced_memory()
                trace_mem_trace = tracemalloc.get_tracemalloc_memory()
                tracemalloc.stop()
                print(f"memtrace: {mem_trace}")
                print(f"tracememtrace: {trace_mem_trace}")
                total_time = end_time - start_time
                printStr = f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds and used up to {mem_trace[1]} bytes of memory'
                if len(printStr) > 200:
                    printStr = f'Function {func.__name__} Took {total_time:.4f} seconds and used up to {mem_trace[1]} bytes of memory'
                print(printStr)
                print()
                PerfTracker.tracking_data[name]["runtimes"].append(total_time)
                PerfTracker.tracking_data[name]["maxmems"].append(mem_trace[1])
                return result
            return perftrack_wrapper
        return track_decorator
    
    def export_to_csv(name=None):
        if name is not None:
            value = PerfTracker.tracking_data[value]
            funcdata = np.array(value["runtimes"])
            funcdata = np.append(funcdata, np.array(value["maxmems"]))
            if funcdata.shape[0] == 0:
                print(f"No data in perf tracker for {name}")
                return
            pd.DataFrame(funcdata).to_csv(f"perfdata_{key}.csv")
            return
        
        flatteneddata = {}
        for key, value in PerfTracker.tracking_data.items():
            funcdata = np.array(value["runtimes"])
            funcdata = np.append(funcdata, np.array(value["maxmems"]))
            if funcdata.shape[0] != 0:
                pd.DataFrame(funcdata).to_csv(f"perfdata_{key}.csv")
                flatteneddata[f"{key}runtimes"] = value["runtimes"]
                flatteneddata[f"{key}maxmems"] = value["maxmems"]
        if len(flatteneddata) == 0:
            print("PerfTracker has no data to export")
        all_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in flatteneddata.items() ]))
        all_data.to_csv("perfdata_all.csv")
    
    def __str__(self):
        return "%s" % (self._self_str)
    

if __name__ == "__main__":
    @PerfTracker.track("num")
    def test_function_to_track(num: int):
        return num * num + 2

    test_function_to_track(4)