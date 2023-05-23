# ------------------------------------------------------------------------
# DCQ-Det
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .DCQDet import build_DCQ_Det

def build_model(args):
    return build_DCQ_Det(args)
