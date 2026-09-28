import zarr


def print_zarr_tree(path):
    root = zarr.open(path, mode="r")

    def walk(group, prefix=""):
        for name in group:
            obj = group[name]

            if isinstance(obj, zarr.Array):
                print(
                    f"{prefix}{name}: "
                    f"shape={obj.shape}, dtype={obj.dtype}"
                )
            else:
                print(f"{prefix}{name}/")
                walk(obj, prefix + "  ")

    walk(root)

file = "/xfel/ffs/dat/ue_260903_FXS/archive/run00022/scan00001/raw.zarr"
print_zarr_tree(file)