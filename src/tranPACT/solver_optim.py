"""Optimization parameters for optimization-based reconstruction method (OBRM)."""

__all__ = ['OptimParam']

class OptimParam:
    def __init__(self, reg=0.01, num_iter=100, out_print=0, lip=5.0,
                 saving_dir=None, use_check=False, check_timeout=None, 
                 save_freq=1):
        self.reg = reg
        self.num_iter = num_iter
        self.out_print = out_print
        self.lip = lip
        self.saving_dir = saving_dir
        self.use_check = use_check
        self.check_timeout = check_timeout
        self.positive_constraint = True
        self.save_freq = save_freq

    def input_args(self, args):
        """
        Input arguments from command line.
        Args:
            args: The command line arguments.
        """
        for key, value in args.items():
            if hasattr(self, key):  # Check if the attribute exists
                setattr(self, key, value)  # Set the attribute