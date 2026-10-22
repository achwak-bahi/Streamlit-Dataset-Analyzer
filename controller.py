import model


def get_available_datasets():
    return model.list_datasets()


def get_dataset_groups():
    return model.get_dataset_groups()


def get_group_files(group_name):
    return model.get_group_files(group_name)


def get_group_info(group_name):
    return model.get_group_info(group_name)


def get_subsets_summary(group_name):
    return model.get_subsets_summary(group_name)


def get_group_dataframe(group_name, subset_file=None):
    return model.get_group_dataframe(group_name, subset_file=subset_file)


def analyze_group(group_name, subset_file=None):
    return model.analyze_group(group_name, subset_file=subset_file)


def analyze_dataset(dataset_name):
    return model.analyze_dataset(dataset_name)


def analyze_uploaded_file(filename, content):
    return model.analyze_uploaded_file(filename, content)


def get_dataframe(dataset_name=None, filename=None, content=None):
    return model.get_dataframe(dataset_name=dataset_name, filename=filename, content=content)
