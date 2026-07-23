def is_using_cuda():
    import os
    from pathlib import Path
    # Restrict the program to one physical GPU.
    # This must be set before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3")
    cuda_include = os.environ.get("CUDA_INCLUDE_DIR", f"{cuda_home}/targets/sbsa-linux/include")
    cuda_library = os.environ.get("CUDA_LIBRARY_DIR", f"{cuda_home}/targets/sbsa-linux/lib")
    ptxas_path = os.environ.get("TRITON_PTXAS_PATH", f"{cuda_home}/bin/ptxas",)
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["TRITON_PTXAS_PATH"] = ptxas_path

    os.environ["PATH"] = os.pathsep.join(
        value
        for value in [
            str(Path(ptxas_path).parent),
            os.environ.get("PATH"),
        ]
        if value
    )

    os.environ["CPATH"] = os.pathsep.join(
        value
        for value in [
            cuda_include,
            os.environ.get("CPATH"),
        ]
        if value
    )

    os.environ["LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in [
            cuda_library,
            os.environ.get("LIBRARY_PATH"),
        ]
        if value
    )

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in [
            cuda_library,
            os.environ.get("LD_LIBRARY_PATH"),
        ]
        if value
    )