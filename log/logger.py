import logging



logging.basicConfig(level=logging.DEBUG,
                    filename="./log/prova.log",
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)