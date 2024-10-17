def show_detail(
    datasets,
    shape=False,
    column=False,
    info=False,
    describe=False,
    is_null=False,
    dtype=False,
):
    for name, df in datasets:
        if shape:
            print(f"{name}: {df.shape}")
        #-----------------
        if column:
            print(f"{name}: {df.columns}")
        #-----------------
        if info:
            print(f"{name}:")
            df.info()
            print()
        #-----------------
        if describe:
            print(f"{name}:")
            print(df.describe())
            print()
        #-----------------
        if is_null:
            print(f"{name} missing values:")
            print(df.isnull().sum())
            print()
        #-----------------
        if dtype:
            print(f"{name} data types:")
            print(df.dtypes)
            print()