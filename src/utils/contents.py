COLORS = {'black': '\033[30m',  # basic colors
          'red': '\033[31m',
          'green': '\033[32m',
          'yellow': '\033[33m',
          'blue': '\033[34m',
          'magenta': '\033[35m',
          'cyan': '\033[36m',
          'white': '\033[37m',
          'bright_black': '\033[90m',  # bright colors
          'bright_red': '\033[91m',
          'bright_green': '\033[92m',
          'bright_yellow': '\033[93m',
          'bright_blue': '\033[94m',
          'bright_magenta': '\033[95m',
          'bright_cyan': '\033[96m',
          'bright_white': '\033[97m',
          'end': '\033[0m',  # misc
          'bold': '\033[1m',
          'underline': '\033[4m'}

# 去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
INVALID_DATA = ['ts_code', 'id', 'pre_close', 'trade_date', 'date']
