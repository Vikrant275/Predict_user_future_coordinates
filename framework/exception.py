import sys

class MyException(Exception):
    def __init__(self, error_msg,error_code:sys):
        self.error_msg = error_msg
        _,_,exc_tb = error_code.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occurred in python script name [{0}] line no. [{1}] error message [{2}] ".format(self.filename,self.lineno,str(self.error_msg))

