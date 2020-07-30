from joblib import register_parallel_backend

def register_cloudbutton():
    """ Register Cloudbutton Backend to be called with
        joblib.parallel_backend("cloudbutton") """
    try:
        from cloudbutton.util.joblib.cloudbutton_backend import CloudbuttonBackend
        register_parallel_backend("cloudbutton", CloudbuttonBackend)
    except ImportError:
        msg = ("To use the cloudbutton backend you must first install the plugin. "
               "See https://github.com/Dahk/cloudbutton-backend.git "
               "for instructions.")
        raise ImportError(msg)


__all__ = ["register_cloudbutton"]
