# Keras preprocessing - making it picklable
# The function is run only when keras is necessary
def make_keras_picklable():
    import keras.models
    import tempfile
    cls = keras.models.Model

    if hasattr(cls, "is_now_picklable"):
        return

    cls.is_now_picklable = True

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__