import logging
import logging.config
import logging_tree

config_path = '/home/bugger/logging.ini'
logging.config.fileConfig(config_path, defaults={'logfilename': '/home/bugger/mylog.log'})

logger = logging.getLogger('main')
print(logger)

logger.debug('often makes a very good meal of %s', 'visiting tourists')
logging_tree.printout()