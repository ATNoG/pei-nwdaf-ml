from src.general.modelI import ModelI
import logging

logger = logging.getLogger("Inference")

class InferenceMaker:
    """Cooks an inference"""
    def __init__(self, selected_model_id:str|None = None ) -> None:

        self._auto_mode:bool = False   # whether inference should auto-select best model of use provided model
        self._failed_retrieves:int = 0 # times that couldn't retrieve model from repository

        if selected_model_id is None:
            #TODO: select best model from repository
            self._auto_mode = True


        else:
            #TODO: validate model
            self._current_model_id = selected_model_id


    def _fetch_best_model(self) -> str:
        """Returns the id of the best model from repository"""
        #TODO: find best model on repository
        return ""

    def toggle_auto_select(self,value:bool|None = None):
        """Toggles auto mode"""
        self._auto_mode = value if value is not None else not self._auto_mode

    def _load_model(self) -> ModelI|None:
        """Loads selected model from repository"""
        target_model_id:str = self._fetch_best_model() if self._auto_mode else self._current_model_id
        #TODO: implement logic to load model
        return None

    def _set_model(self, model_id:str) -> bool:
        """
        Sets inference to use a model.
        Args:
            model_id(str) : id of the model
        Returns:
            bool: found model?
        """
        # TODO: check if model exists
        self._current_model_id = model_id
        self._auto_mode = False
        return False

    def infer(self,**args) -> ...:
        """
        Produces an inference using the selected model
        """

        # load model
        model:ModelI|None = self._load_model()
        if model is None:
            logger.warning("Couldn't load model")
            return

        return model.infer(**args)
