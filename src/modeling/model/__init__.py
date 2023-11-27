# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

__version__ = "1.0.1"

from .ignore_modeling_fastmetro import (FastMETRO_Hand_Network)
from .modeling_xyz_fastmetro import (FastMETRO_Body_Network) # SMPL-less, just return X,Y,Z points