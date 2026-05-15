# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

platform = current_platform()

flash_mla_with_kvcache = error_fn
get_mla_metadata = error_fn
flashmla_dense_fp8_fwd = error_fn
flashmla_dense_fp8_metadata = error_fn

if platform.is_nvidia and platform.is_hopper:
    try:
        from flash_mla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except ImportError:
        pass

    # Dense FP8 MLA decode kernel (sm_90a only).
    try:
        from tokenspeed_kernel.thirdparty.cuda.flashmla_dense_fp8 import (
            flashmla_dense_fp8_fwd,
            flashmla_dense_fp8_metadata,
        )
    except ImportError:
        pass

# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

__all__ = [
    "flash_mla_with_kvcache",
    "get_mla_metadata",
    "flashmla_dense_fp8_fwd",
    "flashmla_dense_fp8_metadata",
]
